using OptimizationDynamics
using LinearAlgebra
using Random
using MeshCat
using Colors
using JSON3
using OrderedCollections: OrderedDict
ENV["GKSwstype"] = "100"

const iLQR = OptimizationDynamics.IterativeLQR

# ------------------------------
# Configuration
# ------------------------------
GB = false
SHOW_VIS = true
RUN_DISTURBANCE = true
PLOT_RESULTS = true
PLOT_DIAGNOSTICS = true
SAVE_CSV = false
SOLVER_VERBOSE = false
LINE_PUSH_FREE_BOX_REF_TRAJ_DIR = joinpath(@__DIR__, "..", "data", "reference_trajectory")
LINE_PUSH_FREE_BOX_REF_TRAJ_FILE = joinpath(LINE_PUSH_FREE_BOX_REF_TRAJ_DIR, "line_push_free_box_ref_traj.json")
LINE_PUSH_FREE_DATA_DIR = joinpath(@__DIR__, "..", "data", "line_push_free_data")
LINE_PUSH_FREE_DATA_FILE = joinpath(LINE_PUSH_FREE_DATA_DIR, "line_push_free_rollout_data.json")

h = 0.05
T = 25
num_w = lineplanarpush_xy.nw
nc = 2
nc_impact = 2
r_dim = 0.1
pusher_y_offset = 0.025

x_goal = 0.5
y_goal = 0.2
θ_goal = 1.1

uw_values = [0, 0.001, 0.0025, 0.005] # disturbance values

test_num_w = 3
DISTURBANCE_SCALE = 0.0

# ------------------------------
# Dynamics
# ------------------------------
im_dyn = ImplicitDynamics(
    lineplanarpush_xy,
    h,
    eval(r_lppxy_func),
    eval(rz_lppxy_func),
    eval(rθ_lppxy_func);
    r_tol=1.0e-8,
    κ_eval_tol=1.0e-4,
    κ_grad_tol=1.0e-2,
    nc=2,
    nb=10,
    d=num_w,
    info=(GB ? GradientBundle(lineplanarpush_xy, N=50, ϵ=1.0e-4) : nothing),
)

nx = 2 * lineplanarpush_xy.nq
nu = lineplanarpush_xy.nu

ilqr_dyn = iLQR.Dynamics(
    (d, x, u, w) -> f(d, im_dyn, x, u, w),
    (dx, x, u, w) -> GB ? fx_gb(dx, im_dyn, x, u, w) : fx(dx, im_dyn, x, u, w),
    (du, x, u, w) -> GB ? fu_gb(du, im_dyn, x, u, w) : fu(du, im_dyn, x, u, w),
    (gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dyn, x, u, w),
    nx,
    nx,
    nu,
    num_w,
    nc,
    nc_impact,
)

ilqr_dyns = [ilqr_dyn for _ = 1:T-1]

# ------------------------------
# Initial conditions and goal
# ------------------------------
q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, pusher_y_offset, -r_dim - 1.0e-8, -pusher_y_offset]
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, pusher_y_offset, -r_dim - 1.0e-8, -pusher_y_offset]
qT = [x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal + pusher_y_offset, x_goal - r_dim, y_goal - pusher_y_offset]

x1 = [q0; q1]
xT = [qT; qT]

# ------------------------------
# Objective
# ------------------------------
Qv = Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
Qx = Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
Ru = 0.1
ϕ_weight = 10.0
Wp_move = Diagonal([1.0, 1.0, 1.0, 1.0]) # pusher step-to-step movement penalty
γ_balance_weight = 0.3

# Diagnostic helper: evaluate current normal contact forces.
const _dyn_eval_buf = zeros(nx)
const _empty_dbg_vec = zeros(0)
const _gamma_obj_buf = zeros(nc_impact)
function eval_gamma_contacts!(γ_out, x, u, w)
    f(_dyn_eval_buf, im_dyn, x, u, w)
    f_debug(γ_out, _empty_dbg_vec, _empty_dbg_vec, _empty_dbg_vec, im_dyn, x, u, w)
    return γ_out
end

function eval_contact_data!(γ_out, b_out, x, u, w)
    f(_dyn_eval_buf, im_dyn, x, u, w)
    f_debug(γ_out, b_out, _empty_dbg_vec, _empty_dbg_vec, im_dyn, x, u, w)
    return γ_out, b_out
end

function state_parts(x)
    nq = lineplanarpush_xy.nq
    q1 = @views x[1:nq]
    q2 = @views x[nq .+ (1:nq)]
    v1 = (q2 - q1) ./ h
    return q1, q2, v1
end

function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
end

function rollout_feedback(dyns, x1, x_nom, u_nom, K, w_seq, ul, uu)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = u_nom[t] + K[t] * (x - x_nom[t])
        ut = clamp_u(ut, ul, uu)
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        u_hist[t] = copy(ut)
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

to_float_vec(x) = Float64[v for v in x]
to_float_vecs(xs) = [to_float_vec(x) for x in xs]

function save_line_push_free_data(
    out_file::String;
    h::Float64,
    q_goal,
    q_nom,
    u_nom,
    gamma_nom,
    w_nom,
    q_fb=nothing,
    u_fb=nothing,
    gamma_fb=nothing,
    w_fb=nothing,
)
    goal_pose = q_goal[1:3]

    nom_final = q_nom[end]
    nom_pose_err = nom_final[1:2] - q_goal[1:2]
    nom_theta_err = nom_final[3] - q_goal[3]
    nom_control_effort = sum(dot(u, u) for u in u_nom)

    od = OrderedDict{String,Any}()
    od["h"] = h
    od["goal_pose"] = to_float_vec(goal_pose)

    nominal = OrderedDict{String,Any}()
    nominal["q"] = to_float_vecs(q_nom)
    nominal["u"] = to_float_vecs(u_nom)
    nominal["gamma"] = to_float_vecs(gamma_nom)
    nominal["w"] = to_float_vecs(w_nom)
    nominal["control_effort"] = nom_control_effort
    nominal["pose_error"] = to_float_vec(nom_pose_err)
    nominal["pose_error_norm"] = norm(nom_pose_err)
    nominal["theta_error"] = nom_theta_err
    od["nominal"] = nominal

    closed_loop = OrderedDict{String,Any}()
    if q_fb === nothing || u_fb === nothing || gamma_fb === nothing || w_fb === nothing
        closed_loop["available"] = false
    else
        fb_final = q_fb[end]
        fb_pose_err = fb_final[1:2] - q_goal[1:2]
        fb_theta_err = fb_final[3] - q_goal[3]
        fb_control_effort = sum(dot(u, u) for u in u_fb)
        closed_loop["available"] = true
        closed_loop["q"] = to_float_vecs(q_fb)
        closed_loop["u"] = to_float_vecs(u_fb)
        closed_loop["gamma"] = to_float_vecs(gamma_fb)
        closed_loop["w"] = to_float_vecs(w_fb)
        closed_loop["control_effort"] = fb_control_effort
        closed_loop["pose_error"] = to_float_vec(fb_pose_err)
        closed_loop["pose_error_norm"] = norm(fb_pose_err)
        closed_loop["theta_error"] = fb_theta_err
    end
    od["closed_loop"] = closed_loop

    mkpath(dirname(out_file))
    open(out_file, "w") do io
        JSON3.write(io, od; indent=2)
    end
    return nothing
end

function objt(x, u, w)
    q1, q2, v1 = state_parts(x)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)
    J += 0.5 * Ru * transpose(u) * u
    Δp = q2[4:7] - q1[4:7]
    J += 0.5 * transpose(Δp) * Wp_move * Δp

    ϕ = ϕ_func(lineplanarpush_xy, q2)
    J += 0.5 * ϕ_weight * ϕ[1]^2
    J += 0.5 * ϕ_weight * ϕ[2]^2
    # Encourage both contacts to share load without hard-enforcing min normal force.
    if x isa Vector{Float64} && u isa Vector{Float64} && w isa Vector{Float64}
        γ = eval_gamma_contacts!(_gamma_obj_buf, x, u, w)
        J += 0.5 * γ_balance_weight * (γ[1] - γ[2])^2
    end

    return J
end

function objT(x, u, w)
    _, q2, v1 = state_parts(x)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)

    ϕ = ϕ_func(lineplanarpush_xy, q2)
    J += 0.5 * ϕ[1]^2
    J += 0.5 * ϕ[2]^2

    return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for _ = 1:T-1]..., cT]

# ------------------------------
# Constraints
# ------------------------------
ul = [-2.0; -2.0; -2.0; -2.0]
uu = [2.0; 2.0; 2.0; 2.0]
max_pusher_gap = 0.0001
max_tangent_slip_vel = 0.003

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function stage_con(x, u, w)
    q1, q2, _ = state_parts(x)
    ϕ = ϕ_func(lineplanarpush_xy, q2)
    p_block1 = q1[1:2]
    p_block2 = q2[1:2]
    p11_local = transpose(rot2(q1[3])) * (q1[4:5] - p_block1)
    p12_local = transpose(rot2(q1[3])) * (q1[6:7] - p_block1)
    p21_local = transpose(rot2(q2[3])) * (q2[4:5] - p_block2)
    p22_local = transpose(rot2(q2[3])) * (q2[6:7] - p_block2)
    slip_vel1 = (p21_local[2] - p11_local[2]) / h
    slip_vel2 = (p22_local[2] - p12_local[2]) / h

    [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
        ϕ[1] - max_pusher_gap; # keep pusher 1 near the box
        ϕ[2] - max_pusher_gap; # keep pusher 2 near the box
        slip_vel1 - max_tangent_slip_vel;
        -slip_vel1 - max_tangent_slip_vel;
        slip_vel2 - max_tangent_slip_vel;
        -slip_vel2 - max_tangent_slip_vel;
    ]
end

function terminal_con(x, u, w)
    [
        (x - xT)[[8, 9, 10]]; # block x, y, θ
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu + 6)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for _ = 1:T-1]..., conT]

# ------------------------------
# Rollout
# ------------------------------
function initial_control(t)
    if t < 5
        return [0.5; 0.0; 0.5; 0.0]
    elseif t < 10
        return [0.5; 0.0; 0.5; 0.0]
    elseif t < 20
        return [0.05; 0.0; 0.05; 0.0]
    else
        return [0.05; 0.05; 0.05; 0.05]
    end
end

ū = [initial_control(t) for t = 1:T-1]

w = [[DISTURBANCE_SCALE * rand() * rand([-1, 1])] for _ = 1:T]

x̄, _ = iLQR.rollout(ilqr_dyns, x1, ū, w)

# ------------------------------
# Solver
# ------------------------------
solver = iLQR.solver(
    ilqr_dyns,
    obj,
    cons,
    opts=iLQR.Options(
        linesearch=:armijo,
        α_min=1.0e-5,
        obj_tol=1.0e-3,
        grad_tol=1.0e-3,
        max_iter=20,
        max_al_iter=30,
        con_tol=0.005,
        ρ_init=1.0,
        ρ_scale=10.0,
        verbose=SOLVER_VERBOSE,
    ),
)

iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄)

solver.m_data.w = w

iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver)

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost

# ------------------------------
# Solution
# ------------------------------
x_sol, u_sol = iLQR.get_trajectory(solver)
# Re-rollout with optimized controls to get consistent contact force history.
x_eval, gamma_sol = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
q_sol = state_to_configuration(x_eval)

box_goal = qT[1:2]
box_final = q_sol[end][1:2]
box_pos_err = box_final - box_goal
θ_goal_eval = qT[3]
θ_final = q_sol[end][3]
θ_err = θ_final - θ_goal_eval
control_effort = sum(dot(u, u) for u in u_sol)
gamma_comp(γ, i) = (γ isa AbstractVector && length(γ) >= i) ? γ[i] : 0.0
gamma1_vals = [gamma_comp(γ, 1) for γ in gamma_sol]
gamma2_vals = [gamma_comp(γ, 2) for γ in gamma_sol]
gamma_diff_vals = gamma1_vals - gamma2_vals
gamma1_mean_abs = sum(abs.(gamma1_vals)) / length(gamma1_vals)
gamma2_mean_abs = sum(abs.(gamma2_vals)) / length(gamma2_vals)
gamma_diff_peak = maximum(abs.(gamma_diff_vals))
gamma_eps = 1.0e-4
gamma1_min = minimum(gamma1_vals)
gamma1_max = maximum(gamma1_vals)
gamma2_min = minimum(gamma2_vals)
gamma2_max = maximum(gamma2_vals)
gamma1_active_ratio = count(>(gamma_eps), gamma1_vals) / length(gamma1_vals)
gamma2_active_ratio = count(>(gamma_eps), gamma2_vals) / length(gamma2_vals)
tau_proxy_hist = Float64[]
slip_vel1_hist = Float64[]
slip_vel2_hist = Float64[]
for t in 1:length(u_sol)
    q_prev = q_sol[t]
    q = q_sol[t + 1]
    p_block_prev = q_prev[1:2]
    p_block = q[1:2]
    Rwb_prev = rot2(q_prev[3])
    Rwb = rot2(q[3])
    p1_local_prev = transpose(Rwb_prev) * (q_prev[4:5] - p_block_prev)
    p2_local_prev = transpose(Rwb_prev) * (q_prev[6:7] - p_block_prev)
    p1_local = transpose(Rwb) * (q[4:5] - p_block)
    p2_local = transpose(Rwb) * (q[6:7] - p_block)
    γt1 = gamma_comp(gamma_sol[t], 1)
    γt2 = gamma_comp(gamma_sol[t], 2)
    push!(tau_proxy_hist, -(p1_local[2] * γt1 + p2_local[2] * γt2))
    push!(slip_vel1_hist, (p1_local[2] - p1_local_prev[2]) / h)
    push!(slip_vel2_hist, (p2_local[2] - p2_local_prev[2]) / h)
end
tau_proxy_peak = maximum(abs.(tau_proxy_hist))
slip1_max_abs = maximum(abs.(slip_vel1_hist))
slip2_max_abs = maximum(abs.(slip_vel2_hist))
slip_max_abs = max(slip1_max_abs, slip2_max_abs)
slip_margin_to_bound = max_tangent_slip_vel - slip_max_abs
@show box_goal
@show box_final
@show box_pos_err
@show norm(box_pos_err)
@show θ_goal_eval
@show θ_final
@show θ_err
@show control_effort
@show gamma1_mean_abs
@show gamma2_mean_abs
@show gamma_diff_peak
@show gamma1_min
@show gamma1_max
@show gamma2_min
@show gamma2_max
@show gamma1_active_ratio
@show gamma2_active_ratio
@show tau_proxy_peak
@show slip1_max_abs
@show slip2_max_abs
@show slip_margin_to_bound

if PLOT_DIAGNOSTICS
    using Plots
    time_states = collect(0:h:(length(q_sol) - 1) * h)
    time_controls = collect(0:h:(length(u_sol) - 1) * h)
    θ_trace = [q[3] for q in q_sol]
    n_state = minimum((length(time_states), length(θ_trace)))
    t_state = time_states[1:n_state]
    θ_plot = θ_trace[1:n_state]
    θ_goal_line = fill(θ_goal_eval, n_state)
    u_norm_hist = [norm(u) for u in u_sol]
    n_ctrl = minimum((length(time_controls), length(gamma1_vals), length(gamma2_vals), length(gamma_diff_vals), length(tau_proxy_hist), length(u_norm_hist)))
    t_ctrl = time_controls[1:n_ctrl]
    gamma1_plot = gamma1_vals[1:n_ctrl]
    gamma2_plot = gamma2_vals[1:n_ctrl]
    gamma_diff_plot = gamma_diff_vals[1:n_ctrl]
    tau_plot = tau_proxy_hist[1:n_ctrl]
    u_norm_plot = u_norm_hist[1:n_ctrl]
    n_slip = minimum((length(time_controls), length(slip_vel1_hist), length(slip_vel2_hist)))
    t_slip = time_controls[1:n_slip]
    slip1_plot = slip_vel1_hist[1:n_slip]
    slip2_plot = slip_vel2_hist[1:n_slip]

    p1 = plot(t_state, θ_plot, label="theta", linewidth=2, color=:blue)
    plot!(p1, t_state, θ_goal_line, label="theta_goal", linewidth=2, color=:black, linestyle=:dash)
    savefig(p1, "data/line_diag_theta.png")

    p2 = plot(t_ctrl, gamma1_plot, label="gamma1", linewidth=2, color=:green)
    plot!(p2, t_ctrl, gamma2_plot, label="gamma2", linewidth=2, color=:olive)
    plot!(p2, t_ctrl, gamma_diff_plot, label="gamma1-gamma2", linewidth=2, color=:orange)
    plot!(p2, t_ctrl, tau_plot, label="tau_proxy", linewidth=2, color=:red)
    plot!(p2, t_ctrl, u_norm_plot, label="u_norm", linewidth=2, color=:blue)
    savefig(p2, "data/line_diag_force_tau_u.png")

    gamma_sum_plot = gamma1_plot .+ gamma2_plot
    p_gamma = plot(t_ctrl, gamma1_plot, label="gamma1", linewidth=2, color=:green)
    plot!(p_gamma, t_ctrl, gamma2_plot, label="gamma2", linewidth=2, color=:olive)
    plot!(p_gamma, t_ctrl, gamma_sum_plot, label="gamma_sum", linewidth=2, color=:black, linestyle=:dash)
    title!(p_gamma, "Line Contact Normal Force (gamma)")
    xlabel!(p_gamma, "Time (s)")
    ylabel!(p_gamma, "Normal force")
    savefig(p_gamma, "data/line_diag_gamma.png")

    slip_ub = fill(max_tangent_slip_vel, n_slip)
    slip_lb = fill(-max_tangent_slip_vel, n_slip)
    p3 = plot(t_slip, slip1_plot, label="slip_vel1", linewidth=2, color=:magenta)
    plot!(p3, t_slip, slip2_plot, label="slip_vel2", linewidth=2, color=:purple)
    plot!(p3, t_slip, slip_ub, label="slip_ub", linewidth=2, color=:black, linestyle=:dash)
    plot!(p3, t_slip, slip_lb, label="slip_lb", linewidth=2, color=:black, linestyle=:dash)
    savefig(p3, "data/line_diag_slip.png")
end

# ------------------------------
# Optional: disturbance evaluation
# ------------------------------
if RUN_DISTURBANCE && num_w > 0
    Random.seed!(1234)
    uw = [[(uw_values[test_num_w] + 0.01 * rand()) * rand([-1, 1])] for _ = 1:T]

    _, gamma_actual = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
    x_dist_open, gamma_hist_dist_open = iLQR.rollout(ilqr_dyns, x1, u_sol, uw)
    q_dist_open = state_to_configuration(x_dist_open)

    K = solver.p_data.K
    x_dist_fb, u_dist_fb = rollout_feedback(ilqr_dyns, x1, x_sol, u_sol, K, uw, ul, uu)
    q_dist_fb = state_to_configuration(x_dist_fb)
    _, gamma_hist_dist_fb = iLQR.rollout(ilqr_dyns, x1, u_dist_fb, uw)

    open_box_final = q_dist_open[end][1:2]
    fb_box_final = q_dist_fb[end][1:2]
    open_theta_final = q_dist_open[end][3]
    fb_theta_final = q_dist_fb[end][3]
    @show norm(open_box_final - qT[1:2])
    @show norm(fb_box_final - qT[1:2])
    @show abs(open_theta_final - qT[3])
    @show abs(fb_theta_final - qT[3])
end

mkpath(LINE_PUSH_FREE_DATA_DIR)
if RUN_DISTURBANCE && num_w > 0 && @isdefined(q_dist_fb) && @isdefined(u_dist_fb) && @isdefined(gamma_hist_dist_fb) && @isdefined(uw)
    save_line_push_free_data(
        LINE_PUSH_FREE_DATA_FILE;
        h=h,
        q_goal=qT,
        q_nom=q_sol,
        u_nom=u_sol,
        gamma_nom=gamma_sol,
        w_nom=w,
        q_fb=q_dist_fb,
        u_fb=u_dist_fb,
        gamma_fb=gamma_hist_dist_fb,
        w_fb=uw,
    )
else
    save_line_push_free_data(
        LINE_PUSH_FREE_DATA_FILE;
        h=h,
        q_goal=qT,
        q_nom=q_sol,
        u_nom=u_sol,
        gamma_nom=gamma_sol,
        w_nom=w,
    )
end
println("saved rollout data: " * LINE_PUSH_FREE_DATA_FILE)

# ------------------------------
# Optional: plotting
# ------------------------------
if PLOT_RESULTS && RUN_DISTURBANCE && num_w > 0
    using Plots

    θ_sol = [q_sol[i][3] for i in 1:T]
    θ_dist_open = [q_dist_open[i][3] for i in 1:T]
    θ_dist_fb = [q_dist_fb[i][3] for i in 1:T]
    time = collect(0:h:(T-1) * h)
    θ_goal_line = fill(θ_goal, length(time))

    plot(time, θ_sol, label="actual_θ", linewidth=2, color=:green)
    plot!(time, θ_dist_open, label="dist_open_θ", linewidth=2, color=:red)
    plot!(time, θ_dist_fb, label="dist_fb_θ", linewidth=2, color=:blue)
    plot!(time, θ_goal_line, label="goal_θ", linewidth=2, color=:black, linestyle=:dash)
    title!("[line_xy] θ with dist (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Rotation (rad)")
    savefig("data/line_xy_θ_goal_$(θ_goal)_$(uw_values[test_num_w]).png")

    gamma_sol_vals = [gamma_comp(gamma_actual[i], 1) for i in 1:T-1]
    gamma_open_vals = [gamma_comp(gamma_hist_dist_open[i], 1) for i in 1:T-1]
    gamma_open_vals2 = [gamma_comp(gamma_hist_dist_open[i], 2) for i in 1:T-1]
    gamma_fb_vals = [gamma_comp(gamma_hist_dist_fb[i], 1) for i in 1:T-1]
    gamma_fb_vals2 = [gamma_comp(gamma_hist_dist_fb[i], 2) for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, gamma_sol_vals, label="γ_nom", linewidth=2, color=:green)
    plot!(time_controls, gamma_open_vals .+ gamma_open_vals2, label="γ_dist_open_sum", linewidth=2, color=:red)
    plot!(time_controls, gamma_fb_vals .+ gamma_fb_vals2, label="γ_dist_fb_sum", linewidth=2, color=:blue)
    title!("[line_xy] Contact Force (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Contact Force")
    savefig("data/line_xy_contact_force_$(θ_goal)_$(uw_values[test_num_w]).png")

    u1_vals = [u_sol[i][1] for i in 1:T-1]
    u2_vals = [u_sol[i][2] for i in 1:T-1]
    u3_vals = [u_sol[i][3] for i in 1:T-1]
    u4_vals = [u_sol[i][4] for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, u1_vals, label="u_x1", linewidth=2, color=:blue)
    plot!(time_controls, u2_vals, label="u_y1", linewidth=2, color=:green)
    plot!(time_controls, u3_vals, label="u_x2", linewidth=2, color=:blue, linestyle=:dash)
    plot!(time_controls, u4_vals, label="u_y2", linewidth=2, color=:green, linestyle=:dash)
    title!("[line_xy] Control Inputs (θ_goal=$(θ_goal))")
    xlabel!("Time (s)")
    ylabel!("Control Input")
    savefig("data/line_xy_control_inputs_$(θ_goal).png")
end

# ------------------------------
# Optional: visualization
# ------------------------------
if SHOW_VIS
    vis = Visualizer()
    render(vis)
    if RUN_DISTURBANCE && num_w > 0 && @isdefined(q_dist_fb)
        q_nom_vis = q_sol
        q_dist_vis = q_dist_fb
        Tvis = min(length(q_nom_vis), length(q_dist_vis))

        OptimizationDynamics.default_background!(vis)
        # i=1: nominal (transparent), i=2: disturbed (opaque)
        OptimizationDynamics._create_planar_push!(
            vis,
            lineplanarpush_xy,
            i=1,
            tl=0.35,
            box_color=RGBA(0.15, 0.45, 0.95, 0.30),
            pusher_color=RGBA(0.15, 0.75, 0.95, 0.35),
        )
        OptimizationDynamics._create_planar_push!(
            vis,
            lineplanarpush_xy,
            i=2,
            tl=0.95,
            box_color=RGBA(0.95, 0.25, 0.20, 0.95),
            pusher_color=RGBA(0.95, 0.60, 0.20, 0.95),
        )

        anim = MeshCat.Animation(convert(Int, floor(1.0 / h)))
        for t in 1:(Tvis - 1)
            MeshCat.atframe(anim, t) do
                OptimizationDynamics._set_planar_push!(vis, lineplanarpush_xy, q_nom_vis[t], i=1)
                OptimizationDynamics._set_planar_push!(vis, lineplanarpush_xy, q_dist_vis[t], i=2)
            end
        end

        settransform!(vis["/Cameras/default"],
            OptimizationDynamics.compose(
                OptimizationDynamics.Translation(0.0, 0.0, 50.0),
                OptimizationDynamics.LinearMap(
                    OptimizationDynamics.RotZ(0.5 * pi) * OptimizationDynamics.RotY(-pi / 2.5),
                ),
            ))
        setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
        MeshCat.setanimation!(vis, anim)
        println("visualization: nominal=blue/cyan transparent, disturbed-feedback=red/orange opaque")
    else
        visualize!(vis, lineplanarpush_xy, q_sol, Δt=h)
    end
end

# ------------------------------
# Optional: CSV export
# ------------------------------
if SAVE_CSV
    using CSV
    using DataFrames

    function save_to_csv(q_sol, u_sol, T;
        filename_q="data/qdist_without.csv",
        filename_u="data/udist_without.csv"
    )
        nq = length(q_sol[1])
        q_data = DataFrame([getindex.(q_sol, i) for i in 1:nq], ["q_$i" for i in 1:nq])
        CSV.write(filename_q, q_data)

        nu = length(u_sol[1])
        u_data = DataFrame([getindex.(u_sol, i) for i in 1:nu], ["u_$i" for i in 1:nu])
        CSV.write(filename_u, u_data)
    end

    save_to_csv(q_sol, u_sol, T)
end
