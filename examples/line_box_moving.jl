using OptimizationDynamics
using LinearAlgebra
using Random
ENV["GKSwstype"] = "100"

const iLQR = OptimizationDynamics.IterativeLQR

# ------------------------------
# Configuration
# ------------------------------
GB = false
SHOW_VIS = true
RUN_DISTURBANCE = false
PLOT_RESULTS = false
PLOT_DIAGNOSTICS = true
SAVE_CSV = false
SOLVER_VERBOSE = false

h = 0.05
T = 26
num_w = lineplanarpush_xy.nw
nc = 2
nc_impact = 2
r_dim = 0.1
pusher_y_offset = 0.025

x_goal = 0.3
y_goal = 0.2
θ_goal = 0.8

uw_values = [0, 0.001, 0.0025, 0.005] # disturbance values

test_num_w = 1
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

function state_parts(x)
    nq = lineplanarpush_xy.nq
    q1 = @views x[1:nq]
    q2 = @views x[nq .+ (1:nq)]
    v1 = (q2 - q1) ./ h
    return q1, q2, v1
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
ul = [-2.5; -2.5; -2.5; -2.5]
uu = [2.5; 2.5; 2.5; 2.5]
max_pusher_gap = 0.0001
max_tangent_slip = 0.005

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function stage_con(x, u, w)
    _, q2, _ = state_parts(x)
    ϕ = ϕ_func(lineplanarpush_xy, q2)
    p_block = q2[1:2]
    Rwb = rot2(q2[3])
    p1_local = transpose(Rwb) * (q2[4:5] - p_block)
    p2_local = transpose(Rwb) * (q2[6:7] - p_block)
    slip1 = p1_local[2] - pusher_y_offset
    slip2 = p2_local[2] + pusher_y_offset

    [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
        ϕ[1] - max_pusher_gap; # keep pusher 1 near the box
        ϕ[2] - max_pusher_gap; # keep pusher 2 near the box
        slip1 - max_tangent_slip;
        -slip1 - max_tangent_slip;
        slip2 - max_tangent_slip;
        -slip2 - max_tangent_slip;
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
        return [0.5; 0.0; 0.75; 0.0]
    elseif t < 10
        return [0.5; 0.0; 0.75; 0.0]
    elseif t < 20
        return [0.05; 0.0; 0.05; 0.0]
    else
        return [0.05; 0.05; 0.05; 0.05]
    end
end

ū = [initial_control(t) for t = 1:T-1]

w = [[DISTURBANCE_SCALE * rand() * rand([-1, 1])] for _ = 1:T]

x̄, gamma_hist = iLQR.rollout(ilqr_dyns, x1, ū, w)

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
        max_iter=10,
        max_al_iter=20,
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
gamma_sol = iLQR.get_contact_force(solver)
q_sol = state_to_configuration(x_sol)

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
tau_proxy_hist = Float64[]
slip1_hist = Float64[]
slip2_hist = Float64[]
for t in 1:length(u_sol)
    q = q_sol[t + 1]
    p_block = q[1:2]
    Rwb = rot2(q[3])
    p1_local = transpose(Rwb) * (q[4:5] - p_block)
    p2_local = transpose(Rwb) * (q[6:7] - p_block)
    γt1 = gamma_comp(gamma_sol[t], 1)
    γt2 = gamma_comp(gamma_sol[t], 2)
    push!(tau_proxy_hist, -(p1_local[2] * γt1 + p2_local[2] * γt2))
    push!(slip1_hist, p1_local[2] - pusher_y_offset)
    push!(slip2_hist, p2_local[2] + pusher_y_offset)
end
tau_proxy_peak = maximum(abs.(tau_proxy_hist))
slip1_max_abs = maximum(abs.(slip1_hist))
slip2_max_abs = maximum(abs.(slip2_hist))
slip_max_abs = max(slip1_max_abs, slip2_max_abs)
slip_margin_to_bound = max_tangent_slip - slip_max_abs
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
    n_slip = minimum((length(time_controls), length(slip1_hist), length(slip2_hist)))
    t_slip = time_controls[1:n_slip]
    slip1_plot = slip1_hist[1:n_slip]
    slip2_plot = slip2_hist[1:n_slip]

    p1 = plot(t_state, θ_plot, label="theta", linewidth=2, color=:blue)
    plot!(p1, t_state, θ_goal_line, label="theta_goal", linewidth=2, color=:black, linestyle=:dash)
    savefig(p1, "data/line_diag_theta.png")

    p2 = plot(t_ctrl, gamma1_plot, label="gamma1", linewidth=2, color=:green)
    plot!(p2, t_ctrl, gamma2_plot, label="gamma2", linewidth=2, color=:olive)
    plot!(p2, t_ctrl, gamma_diff_plot, label="gamma1-gamma2", linewidth=2, color=:orange)
    plot!(p2, t_ctrl, tau_plot, label="tau_proxy", linewidth=2, color=:red)
    plot!(p2, t_ctrl, u_norm_plot, label="u_norm", linewidth=2, color=:blue)
    savefig(p2, "data/line_diag_force_tau_u.png")

    slip_ub = fill(max_tangent_slip, n_slip)
    slip_lb = fill(-max_tangent_slip, n_slip)
    p3 = plot(t_slip, slip1_plot, label="slip1", linewidth=2, color=:magenta)
    plot!(p3, t_slip, slip2_plot, label="slip2", linewidth=2, color=:purple)
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

    x_temp, gamma_actual = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
    x_dist, gamma_hist_dist = iLQR.rollout(ilqr_dyns, x1, u_sol, uw)
    q_dist = state_to_configuration(x_dist)
end

# ------------------------------
# Optional: plotting
# ------------------------------
if PLOT_RESULTS && RUN_DISTURBANCE && num_w > 0
    using Plots

    θ_sol = [q_sol[i][3] for i in 1:T]
    θ_dist = [q_dist[i][3] for i in 1:T]
    time = collect(0:h:(T-1) * h)
    θ_goal_line = fill(θ_goal, length(time))

    plot(time, θ_sol, label="actual_θ", linewidth=2, color=:green)
    plot!(time, θ_dist, label="dist_θ", linewidth=2, color=:red)
    plot!(time, θ_goal_line, label="goal_θ", linewidth=2, color=:black, linestyle=:dash)
    title!("[line_xy] θ with dist (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Rotation (rad)")
    savefig("data/line_xy_θ_goal_$(θ_goal)_$(uw_values[test_num_w]).png")

    gamma_sol_vals = [gamma_comp(gamma_actual[i], 1) for i in 1:T-1]
    gamma_sol_vals2 = [gamma_comp(gamma_actual[i], 2) for i in 1:T-1]
    gamma_hist_dist_vals = [gamma_comp(gamma_hist_dist[i], 1) for i in 1:T-1]
    gamma_hist_dist_vals2 = [gamma_comp(gamma_hist_dist[i], 2) for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, gamma_sol_vals .+ gamma_hist_dist_vals, label="γ_actual", linewidth=2, color=:green)
    plot!(time_controls, gamma_hist_dist_vals .+ gamma_hist_dist_vals2, label="γ_dist", linewidth=2, color=:red)
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
    visualize!(vis, lineplanarpush_xy, q_sol, Δt=h)
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
