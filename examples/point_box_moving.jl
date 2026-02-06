using OptimizationDynamics
using LinearAlgebra
using Random

const iLQR = OptimizationDynamics.IterativeLQR

# ------------------------------
# Configuration
# ------------------------------
GB = false
SHOW_VIS = true
RUN_DISTURBANCE = false
PLOT_RESULTS = false
SAVE_CSV = false

h = 0.05
T = 26
num_w = planarpush.nw
nc = 1
nc_impact = 1
r_dim = 0.1

θ_goal = 0.5
x_goal = 0.3
y_goal = 0.2

uw_values = [0.0, 0.001, 0.0025, 0.005] # disturbance values
test_num_w = 1
DISTURBANCE_SCALE = 0.0

# ------------------------------
# Dynamics
# ------------------------------
im_dyn = ImplicitDynamics(
    planarpush,
    h,
    eval(r_pp_func),
    eval(rz_pp_func),
    eval(rθ_pp_func);
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

function state_parts(x)
    nq = planarpush.nq
    q1 = @views x[1:nq]
    q2 = @views x[nq .+ (1:nq)]
    v1 = (q2 - q1) ./ h
    return q1, q2, v1
end

function objt(x, u, w)
    _, q2, v1 = state_parts(x)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)
    J += 0.5 * Ru * transpose(u) * u

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
ul = [-5.0; -5.0]
uu = [5.0; 5.0]

function stage_con(x, u, w)
    [
        ul - u; # control limit (lower)
        u - uu; # control limit (upper)
    ]
end

function terminal_con(x, u, w)
    [
        (x - xT)[[1, 2, 3, 6, 7, 8]]; # block x, y, θ
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
    title!("[planar] θ with dist (θ_goal=$(θ_goal), uw=$(uw_values[test_num_w]))")
    xlabel!("Time (s)")
    ylabel!("Rotation (rad)")
    savefig("data/planar_θ_goal_$(θ_goal)_$(uw_values[test_num_w]).png")

    gamma_sol_vals = [gamma_actual[i][1] for i in 1:T-1]
    gamma_hist_dist_vals = [gamma_hist_dist[i][1] for i in 1:T-1]
    time_controls = collect(0:h:(T-2) * h)

    plot(time_controls, gamma_sol_vals .+ gamma_hist_dist_vals, label="γ_actual", linewidth=2, color=:green)
    plot!(time_controls, gamma_hist_dist_vals, label="γ_dist", linewidth=2, color=:red)
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
    visualize!(vis, planarpush, q_sol, Δt=h)
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
