using OptimizationDynamics
using LinearAlgebra
using Random

const iLQR = OptimizationDynamics.IterativeLQR

# ------------------------------
# Configuration
# ------------------------------
GB = false
SHOW_VIS = get(ENV, "LINE_PUSH_FIXED_BOX_SHOW_VIS", "true") == "true"
RUN_DISTURBANCE = get(ENV, "LINE_PUSH_FIXED_BOX_RUN_DISTURBANCE", "false") == "true"
PLOT_RESULTS = get(ENV, "LINE_PUSH_FIXED_BOX_PLOT_RESULTS", "true") == "true"
SAVE_CSV = get(ENV, "LINE_PUSH_FIXED_BOX_SAVE_CSV", "false") == "true"
LINE_PUSH_FIXED_BOX_REF_TRAJ_DIR = joinpath(@__DIR__, "..", "data", "reference_trajectory")
LINE_PUSH_FIXED_BOX_REF_TRAJ_FILE = joinpath(LINE_PUSH_FIXED_BOX_REF_TRAJ_DIR, "line_push_fixed_box_ref_traj.json")

h = 0.05
T = 26
num_w = 1
nc = 2
nc_impact = 2
r_dim = 0.1

rot_goal = [0.17453, 0.17453 * 2, 0.17453 * 3, 0.17453 * 4] # ~= 10deg, 20deg, 30deg
uw_values = [0, 0.001, 0.0025, 0.005] # torque disturbance values

test_number = 1

test_num_w = 1
DISTURBANCE_SCALE = 0.0

# ------------------------------
# Dynamics
# ------------------------------
im_dyn = ImplicitDynamics(
    lineplanarpush,
    h,
    eval(r_lpp_func),
    eval(rz_lpp_func),
    eval(rθ_lpp_func);
    r_tol=1.0e-8,
    κ_eval_tol=1.0e-4,
    κ_grad_tol=1.0e-2,
    nc=2,
    nb=10,
    d=num_w,
    info=(GB ? GradientBundle(lineplanarpush, N=50, ϵ=1.0e-4) : nothing),
)

nx = 2 * lineplanarpush.nq
nu = lineplanarpush.nu

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
q0 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025]
q1 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025]
θ_goal = rot_goal[test_number]
qT = [θ_goal, -r_dim, -r_dim, -r_dim, -r_dim]
xT = zeros(2 * lineplanarpush.nq)
xT[1] = θ_goal
xT[lineplanarpush.nq + 1] = θ_goal

x1 = [q0; q1]

# ------------------------------
# Objective
# ------------------------------
Qv = Diagonal([1.0, 0.0, 0.0, 0.0, 0.0])
Qx = Diagonal([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
Ru = 0.1
ϕ_weight = 200.0
slip_weight = 1.0

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function state_parts(x)
    nq = lineplanarpush.nq
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

    ϕ = ϕ_func(lineplanarpush, q2)
    J += 0.5 * ϕ_weight * ϕ[1]^2
    J += 0.5 * ϕ_weight * ϕ[2]^2
    p11_local = transpose(rot2(q1[1])) * q1[2:3]
    p12_local = transpose(rot2(q1[1])) * q1[4:5]
    p21_local = transpose(rot2(q2[1])) * q2[2:3]
    p22_local = transpose(rot2(q2[1])) * q2[4:5]
    slip_vel1 = (p21_local[2] - p11_local[2]) / h
    slip_vel2 = (p22_local[2] - p12_local[2]) / h
    J += 0.5 * slip_weight * slip_vel1^2
    J += 0.5 * slip_weight * slip_vel2^2

    return J
end

function objT(x, u, w)
    _, q2, v1 = state_parts(x)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)

    ϕ = ϕ_func(lineplanarpush, q2)
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

function stage_con(x, u, w)
    [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
    ]
end

function terminal_con(x, u, w)
    [
        (x - xT)[[6]]; # goal
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu)))
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
        return [0.00; 0.0; 0.00; 0.0]
    else
        return [0.00; 0.00; 0.00; 0.00]
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
        verbose=false,
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

# ------------------------------
# Optional: disturbance evaluation
# ------------------------------
if RUN_DISTURBANCE
    Random.seed!(1234)
    uw = [[(uw_values[test_num_w] + 0.01 * rand()) * rand([-1, 1])] for _ = 1:T]

    x_temp, gamma_actual = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
    x_dist, gamma_hist_dist = iLQR.rollout(ilqr_dyns, x1, u_sol, uw)
    q_dist = state_to_configuration(x_dist)
end

# ------------------------------
# Optional: plotting
# ------------------------------
if PLOT_RESULTS && RUN_DISTURBANCE
    using Plots

    θ_sol = [q_sol[i][1] for i in 1:T]
    θ_dist = [q_dist[i][1] for i in 1:T]
    time = collect(0:h:(T-1) * h)
    θ_goal_line = fill(θ_goal, length(time))

    plot(time, θ_sol, label="actual_θ", linewidth=2, color=:green)
    plot!(time, θ_dist, label="dist_θ", linewidth=2, color=:red)
    plot!(time, θ_goal_line, label="goal_θ", linewidth=2, color=:black, linestyle=:dash)
    title!("[line] θ with dist (θ_goal=$(rot_goal[test_number]), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Rotation (rad)")
    savefig("data/line_θ_goal_$(rot_goal[test_number])_$(uw_values[test_num_w]).png")

    gamma_sol_vals = [gamma_actual[i][1] for i in 1:T-1]
    gamma_sol_vals2 = [gamma_actual[i][2] for i in 1:T-1]
    gamma_hist_dist_vals = [gamma_hist_dist[i][1] for i in 1:T-1]
    gamma_hist_dist_vals2 = [gamma_hist_dist[i][2] for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    # NOTE: This is an instantaneous (per-time-step) sum, not a cumulative sum over time.
    plot(time_controls, gamma_sol_vals .+ gamma_hist_dist_vals, label="γ_actual_sum", linewidth=2, color=:green)
    plot!(time_controls, gamma_hist_dist_vals .+ gamma_hist_dist_vals2, label="γ_dist_sum", linewidth=2, color=:red)
    title!("[line] Contact Force (θ_goal=$(rot_goal[test_number]), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Contact Force")
    savefig("data/line_contact_force_$(rot_goal[test_number])_$(uw_values[test_num_w]).png")

    u1_vals = [u_sol[i][1] for i in 1:T-1]
    u2_vals = [u_sol[i][2] for i in 1:T-1]
    u3_vals = [u_sol[i][3] for i in 1:T-1]
    u4_vals = [u_sol[i][4] for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, u1_vals, label="u_x1", linewidth=2, color=:blue)
    plot!(time_controls, u2_vals, label="u_y1", linewidth=2, color=:green)
    plot!(time_controls, u3_vals, label="u_x2", linewidth=2, color=:blue, linestyle=:dash)
    plot!(time_controls, u4_vals, label="u_y2", linewidth=2, color=:green, linestyle=:dash)
    title!("[line] Control Inputs (θ_goal=$(rot_goal[test_number]))")
    xlabel!("Time (s)")
    ylabel!("Control Input")
    savefig("data/line_control_inputs_$(rot_goal[test_number]).png")
end

# ------------------------------
# Optional: visualization
# ------------------------------
if SHOW_VIS
    vis = Visualizer()
    render(vis)
    visualize!(vis, lineplanarpush, q_sol, Δt=h)
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
