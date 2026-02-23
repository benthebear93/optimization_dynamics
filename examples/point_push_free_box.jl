using OptimizationDynamics
using LinearAlgebra
using Random
using MeshCat
using Colors
using JLD2
using Scratch
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
PLOT_RESULTS = false
PLOT_DIAGNOSTICS = true
SAVE_CSV = false
SOLVER_VERBOSE = false
POINT_PUSH_FREE_BOX_REF_TRAJ_DIR = joinpath(@__DIR__, "..", "data", "reference_trajectory")
POINT_PUSH_FREE_BOX_REF_TRAJ_FILE = joinpath(POINT_PUSH_FREE_BOX_REF_TRAJ_DIR, "point_push_free_box_ref_traj.json")
POINT_PUSH_FREE_DATA_DIR = joinpath(@__DIR__, "..", "data", "point_push_free_data")
POINT_PUSH_FREE_DATA_FILE = joinpath(POINT_PUSH_FREE_DATA_DIR, "point_push_free_rollout_data.json")

h = 0.05
T = 25
num_w = planarpush.nw
nc = 1
nc_impact = 1
r_dim = 0.1

x_goal = 0.5
y_goal = 0.2
θ_goal = 1.1

uw_values = [0.0, 0.001, 0.0025, 0.005] # disturbance values
test_num_w = 3
DISTURBANCE_SCALE = 0.0

function load_planarpush_residuals_from_scratch()
    path = @get_scratch!("planarpush")
    residual_file = joinpath(path, "residual.jld2")
    isfile(residual_file) || error("missing residual file: $(residual_file). Run include(\"src/models/planar_push/codegen.jl\") first.")
    @load residual_file r_pp_func rz_pp_func rθ_pp_func
    return eval(r_pp_func), eval(rz_pp_func), eval(rθ_pp_func)
end

# ------------------------------
# Dynamics
# ------------------------------
r_func, rz_func, rθ_func = load_planarpush_residuals_from_scratch()

im_dyn = ImplicitDynamics(
    planarpush,
    h,
    r_func,
    rz_func,
    rθ_func;
    r_tol=1.0e-8,
    κ_eval_tol=1.0e-4,
    κ_grad_tol=1.0e-2,
    nc=1,
    nb=9,
    d=num_w,
    info=(GB ? GradientBundle(planarpush, N=50, ϵ=1.0e-4) : nothing),
)

nx = 2 * planarpush.nq
nu = planarpush.nu

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
q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
qT = [x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal - r_dim]
xT = [qT; qT]

x1 = [q0; q1]

# ------------------------------
# Objective
# ------------------------------
Qv = Diagonal([1.0, 1.0, 1.0, 0.1, 0.1])
Qx = Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1])
Ru = 0.1
ϕ_weight = 10.0
Wp_move = Diagonal([1.0, 1.0]) # pusher step-to-step movement penalty

# Diagnostic helper: evaluate normal force and contact tangential impulses.
const _dyn_eval_buf = zeros(nx)
const _empty_dbg_vec = zeros(0)
function eval_contact_data!(γ_out, b_out, x, u, w)
    f(_dyn_eval_buf, im_dyn, x, u, w)
    f_debug(γ_out, b_out, _empty_dbg_vec, _empty_dbg_vec, im_dyn, x, u, w)
    return γ_out, b_out
end

function state_parts(x)
    nq = planarpush.nq
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

function save_point_push_free_data(
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
    Δp = q2[4:5] - q1[4:5]
    J += 0.5 * transpose(Δp) * Wp_move * Δp

    ϕ = ϕ_func(planarpush, q2)
    J += 0.5 * ϕ_weight * ϕ[1]^2

    return J
end

function objT(x, u, w)
    _, q2, v1 = state_parts(x)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)

    ϕ = ϕ_func(planarpush, q2)
    J += 0.5 * ϕ[1]^2

    return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for _ = 1:T-1]..., cT]

# ------------------------------
# Constraints
# ------------------------------
ul = [-4.0; -4.0]
uu = [4.0; 4.0]
max_pusher_gap = 0.0001
max_tangent_slip_vel = 0.003

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function stage_con(x, u, w)
    q1, q2, _ = state_parts(x)
    ϕ = ϕ_func(planarpush, q2)
    p_block1 = q1[1:2]
    p_block2 = q2[1:2]
    p1_local = transpose(rot2(q1[3])) * (q1[4:5] - p_block1)
    p2_local = transpose(rot2(q2[3])) * (q2[4:5] - p_block2)
    slip_vel = (p2_local[2] - p1_local[2]) / h

    [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
        ϕ[1] - max_pusher_gap; # keep pusher near the box (no large separation)
        slip_vel - max_tangent_slip_vel;
        -slip_vel - max_tangent_slip_vel;
    ]
end

function terminal_con(x, u, w)
    [
        (x - xT)[[6, 7, 8]]; # block x, y, θ
    ]
end

function write_point_push_free_box_ref_traj_json(
    out_file::String,
    q_sol,
    u_sol,
    w_sol,
    gamma_hist,
    b_hist,
    z_hist,
    theta_hist,
    h::Float64,
    nq::Int,
    nu::Int,
    nw::Int,
)
    H = length(u_sol)
    q = [Vector{Float64}(q_sol[t]) for t in 1:H]
    u = [Vector{Float64}(u_sol[t]) for t in 1:H]
    w = [Vector{Float64}(w_sol[t]) for t in 1:H]
    gamma = [Vector{Float64}(gamma_hist[t]) for t in 1:H]
    b = [Vector{Float64}(b_hist[t]) for t in 1:H]
    z = [Vector{Float64}(z_hist[t]) for t in 1:H]
    theta = [Vector{Float64}(theta_hist[t]) for t in 1:H]

    nc = isempty(gamma) ? 0 : length(gamma[1])
    nb = isempty(b) ? 0 : length(b[1])
    nz = isempty(z) ? 0 : length(z[1])
    ntheta = isempty(theta) ? 0 : length(theta[1])

    iq0 = collect(1:nq)
    iq1 = collect(nq .+ (1:nq))
    iu1 = collect(2 * nq .+ (1:nu))
    iw1 = collect(2 * nq + nu .+ (1:nw))
    iq2 = collect(1:nq)
    igamma1 = collect(nq .+ (1:nc))
    ib1 = collect(nq + nc .+ (1:nb))

    od = OrderedDict{String, Any}()
    od["H"] = H
    od["h"] = h
    od["kappa"] = fill(2.0e-8, H)
    od["q"] = q
    od["u"] = u
    od["w"] = w
    od["gamma"] = gamma
    od["b"] = b
    od["z"] = z
    od["theta"] = theta
    od["iq0"] = iq0
    od["iq1"] = iq1
    od["iu1"] = iu1
    od["iw1"] = iw1
    od["iq2"] = iq2
    od["igamma1"] = igamma1
    od["ib1"] = ib1
    od["nq"] = nq
    od["nu"] = nu
    od["nw"] = nw
    od["nc"] = nc
    od["nb"] = nb
    od["nz"] = nz
    od["nθ"] = ntheta

    open(out_file, "w") do io
        JSON3.write(io, od)
    end
    return nothing
end

function has_nonfinite_nested(vs)
    for v in vs
        for x in v
            if !isfinite(x)
                return true
            end
        end
    end
    return false
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu + 3)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for _ = 1:T-1]..., conT]

# ------------------------------
# Rollout
# ------------------------------
function initial_control(t)
    if t < 5
        return [1.0; 0.0]
    elseif t < 10
        return [1.0; 0.0]
    elseif t < 20
        return [0.1; 0.0]
    else
        return [0.1; 0.1]
    end
end

ū = [initial_control(t) for t = 1:T-1]

w = [zeros(num_w) for _ = 1:T]

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
x_eval, gamma_sol = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
q_sol = state_to_configuration(x_eval)
_, gamma_hist_ref, b_hist_ref, z_hist_ref, theta_hist_ref = iLQR.rollout(ilqr_dyns, x1, u_sol, w)

if has_nonfinite_nested(x_sol) ||
   has_nonfinite_nested(u_sol) ||
   has_nonfinite_nested(q_sol) ||
   has_nonfinite_nested(gamma_hist_ref) ||
   has_nonfinite_nested(b_hist_ref) ||
   has_nonfinite_nested(z_hist_ref) ||
   has_nonfinite_nested(theta_hist_ref)
    error("Non-finite values detected in trajectory. Likely residual codegen is stale after changing planarpush.nw. Re-run include(\"src/models/planar_push/codegen.jl\") and retry.")
end

mkpath(POINT_PUSH_FREE_BOX_REF_TRAJ_DIR)
write_point_push_free_box_ref_traj_json(
    POINT_PUSH_FREE_BOX_REF_TRAJ_FILE,
    q_sol,
    u_sol,
    w,
    gamma_hist_ref,
    b_hist_ref,
    z_hist_ref,
    theta_hist_ref,
    h,
    planarpush.nq,
    planarpush.nu,
    planarpush.nw,
)
println("saved reference json: " * POINT_PUSH_FREE_BOX_REF_TRAJ_FILE)

box_goal = qT[1:2]
box_final = q_sol[end][1:2]
box_pos_err = box_final - box_goal
θ_goal_eval = qT[3]
θ_final = q_sol[end][3]
θ_err = θ_final - θ_goal_eval
control_effort = sum(dot(u, u) for u in u_sol)
gamma_comp(γ, i) = (γ isa AbstractVector && length(γ) >= i) ? γ[i] : 0.0
gamma_vals = [gamma_comp(γ, 1) for γ in gamma_sol]
gamma_peak = maximum(abs.(gamma_vals))
gamma_mean_abs = sum(abs.(gamma_vals)) / length(gamma_vals)
tau_proxy_hist = Float64[]
for t in 1:length(u_sol)
    q = q_sol[t + 1]
    p_local = transpose(rot2(q[3])) * (q[4:5] - q[1:2])
    push!(tau_proxy_hist, -p_local[2] * gamma_vals[t])
end
tau_proxy_peak = maximum(abs.(tau_proxy_hist))
slip_vel_hist = Float64[]
for t in 1:length(u_sol)
    q_prev = q_sol[t]
    q_curr = q_sol[t + 1]
    p_prev = transpose(rot2(q_prev[3])) * (q_prev[4:5] - q_prev[1:2])
    p_curr = transpose(rot2(q_curr[3])) * (q_curr[4:5] - q_curr[1:2])
    push!(slip_vel_hist, (p_curr[2] - p_prev[2]) / h)
end
slip_max_abs = maximum(abs.(slip_vel_hist))
slip_margin_to_bound = max_tangent_slip_vel - slip_max_abs
@show box_goal
@show box_final
@show box_pos_err
@show norm(box_pos_err)
@show θ_goal_eval
@show θ_final
@show θ_err
@show control_effort
@show gamma_mean_abs
@show gamma_peak
@show tau_proxy_peak
@show slip_max_abs
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
    n_ctrl = minimum((length(time_controls), length(gamma_vals), length(tau_proxy_hist), length(u_norm_hist)))
    t_ctrl = time_controls[1:n_ctrl]
    gamma_plot = gamma_vals[1:n_ctrl]
    tau_plot = tau_proxy_hist[1:n_ctrl]
    u_norm_plot = u_norm_hist[1:n_ctrl]

    p1 = plot(t_state, θ_plot, label="theta", linewidth=2, color=:blue)
    plot!(p1, t_state, θ_goal_line, label="theta_goal", linewidth=2, color=:black, linestyle=:dash)
    savefig(p1, "data/point_diag_theta.png")

    p2_gamma = plot(t_ctrl, gamma_plot, label="gamma", linewidth=2, color=:green)
    xlabel!(p2_gamma, "Time (s)")
    ylabel!(p2_gamma, "Contact Force")
    savefig(p2_gamma, "data/point_diag_gamma.png")

    p2 = plot(t_ctrl, gamma_plot, label="gamma", linewidth=2, color=:green)
    plot!(p2, t_ctrl, tau_plot, label="tau_proxy", linewidth=2, color=:red)
    plot!(p2, t_ctrl, u_norm_plot, label="u_norm", linewidth=2, color=:blue)
    savefig(p2, "data/point_diag_force_tau_u.png")

    n_slip = minimum((length(time_controls), length(slip_vel_hist)))
    t_slip = time_controls[1:n_slip]
    slip_plot = slip_vel_hist[1:n_slip]
    slip_ub = fill(max_tangent_slip_vel, n_slip)
    slip_lb = fill(-max_tangent_slip_vel, n_slip)
    p3 = plot(t_slip, slip_plot, label="slip_vel", linewidth=2, color=:magenta)
    plot!(p3, t_slip, slip_ub, label="slip_ub", linewidth=2, color=:black, linestyle=:dash)
    plot!(p3, t_slip, slip_lb, label="slip_lb", linewidth=2, color=:black, linestyle=:dash)
    savefig(p3, "data/point_diag_slip.png")
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

mkpath(POINT_PUSH_FREE_DATA_DIR)
if RUN_DISTURBANCE && num_w > 0 && @isdefined(q_dist_fb) && @isdefined(u_dist_fb) && @isdefined(gamma_hist_dist_fb) && @isdefined(uw)
    save_point_push_free_data(
        POINT_PUSH_FREE_DATA_FILE;
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
    save_point_push_free_data(
        POINT_PUSH_FREE_DATA_FILE;
        h=h,
        q_goal=qT,
        q_nom=q_sol,
        u_nom=u_sol,
        gamma_nom=gamma_sol,
        w_nom=w,
    )
end
println("saved rollout data: " * POINT_PUSH_FREE_DATA_FILE)

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
    title!("[planar] θ with dist (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Rotation (rad)")
    savefig("data/planar_θ_goal_$(θ_goal)_$(uw_values[test_num_w]).png")

    gamma_sol_vals = [gamma_comp(gamma_actual[i], 1) for i in 1:T-1]
    gamma_open_vals = [gamma_comp(gamma_hist_dist_open[i], 1) for i in 1:T-1]
    gamma_fb_vals = [gamma_comp(gamma_hist_dist_fb[i], 1) for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, gamma_sol_vals, label="γ_nom", linewidth=2, color=:green)
    plot!(time_controls, gamma_open_vals, label="γ_dist_open", linewidth=2, color=:red)
    plot!(time_controls, gamma_fb_vals, label="γ_dist_fb", linewidth=2, color=:blue)
    title!("[planar] Contact Force (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Contact Force")
    savefig("data/planar_contact_force_$(θ_goal)_$(uw_values[test_num_w]).png")

    u1_vals = [u_sol[i][1] for i in 1:T-1]
    u2_vals = [u_sol[i][2] for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, u1_vals, label="u_x1", linewidth=2, color=:blue)
    plot!(time_controls, u2_vals, label="u_y1", linewidth=2, color=:green)
    title!("[planar] Control Inputs (θ_goal=$(θ_goal))")
    xlabel!("Time (s)")
    ylabel!("Control Input")
    savefig("data/planar_control_inputs_$(θ_goal).png")
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
            planarpush,
            i=1,
            tl=0.35,
            box_color=RGBA(0.15, 0.45, 0.95, 0.30),
            pusher_color=RGBA(0.15, 0.75, 0.95, 0.35),
        )
        OptimizationDynamics._create_planar_push!(
            vis,
            planarpush,
            i=2,
            tl=0.95,
            box_color=RGBA(0.95, 0.25, 0.20, 0.95),
            pusher_color=RGBA(0.95, 0.60, 0.20, 0.95),
        )

        anim = MeshCat.Animation(convert(Int, floor(1.0 / h)))
        for t in 1:(Tvis - 1)
            MeshCat.atframe(anim, t) do
                OptimizationDynamics._set_planar_push!(vis, planarpush, q_nom_vis[t], i=1)
                OptimizationDynamics._set_planar_push!(vis, planarpush, q_dist_vis[t], i=2)
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
        visualize!(vis, planarpush, q_sol, Δt=h)
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
