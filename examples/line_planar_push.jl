using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

# ## mode
# MODE = :translate
MODE = :rotate 

# ## gradient bundle
GB = false 

h = 0.05
T = 26
num_w = 1
nc_impact = 2
rot_goal = [0.17453, 0.17453*2, 0.17453*3, 0.17453*4] # ~= 10deg, 20deg, 30deg
uw_values = [0.001, 0.0025, 0.005] # torque disturbance values
test_number = 3
test_num_w = 3
r_dim = 0.1
# define implicit dynamics (eval_sim, grad_sim with interior point method)
im_dyn = ImplicitDynamics(lineplanarpush, h, eval(r_lpp_func), eval(rz_lpp_func), eval(rθ_lpp_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-2, nc=2, nb=10, d=num_w, info=(GB ? GradientBundle(lineplanarpush, N=50, ϵ=1.0e-4) : nothing)) 

nx = 2 * lineplanarpush.nq
nu = lineplanarpush.nu 

# ## iLQR from IterativeLQR Pkg - loading user-provided dynamcis
ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
	(dx, x, u, w) -> GB ? fx_gb(dx, im_dyn, x, u, w) : fx(dx, im_dyn, x, u, w), 
	(du, x, u, w) -> GB ? fu_gb(du, im_dyn, x, u, w) : fu(du, im_dyn, x, u, w), 
	(gamma, x, u, w) -> f_debug(gamma, im_dyn, x, u, w),
	nx, nx, nu, num_w, nc_impact) 

# generate ilqr dynamics for each trajectory index
ilqr_dyns = [ilqr_dyn for t = 1:T-1];
# ## initial conditions and goal
if MODE == :translate 
	q0 = [0.0, -r_dim - 1.0e-8, 0.0]
	q1 = [0.0, -r_dim - 1.0e-8, 0.0]
	x_goal = 1.0
	y_goal = 0.0
	θ_goal = 0.0 * π
	qT = [θ_goal,  - r_dim,  - r_dim]
	xT = [qT; qT]
elseif MODE == :rotate 
	q0 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025] #08125
	q1 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025]
	x1 = [q1; q1]
	θ_goal = rot_goal[test_number]
	qT = [θ_goal, -r_dim, -r_dim, -r_dim, -r_dim]
	xT = [qT; qT]
end

# ## objective
function objt(x, u, w)
	J = 0.0 

	q1 = x[1:lineplanarpush.nq] 
	q2 = x[lineplanarpush.nq .+ (1:lineplanarpush.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]) * (x - xT) 
	J += 0.5 * 0.1* transpose(u) * u
	ϕ = ϕ_func(lineplanarpush, q2)[1]  # SDF 값 (접촉 시 0)
	J += 0.5 * 10 * (ϕ)^2
	ϕ = ϕ_func(lineplanarpush, q2)[2]  # SDF 값 (접촉 시 0)
	J += 0.5 * 10 * (ϕ)^2

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:lineplanarpush.nq] 
	q2 = x[lineplanarpush.nq .+ (1:lineplanarpush.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]) * (x - xT) 
	ϕ = ϕ_func(lineplanarpush, q2)[1]  # SDF 값 (접촉 시 0)
	J += 0.5 * (ϕ)^2
	ϕ = ϕ_func(lineplanarpush, q2)[2]  # SDF 값 (접촉 시 0)
	J += 0.5 * (ϕ)^2

	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

# ## constraints
ul = [0.0; -5.0; 0.0; -5.0]
uu = [5.0; 5.0; 5.0; 5.0]

function stage_con(x, u, w) 
    [
     ul - u; # control limit (lower)
     u - uu; # control limit (upper)
    ]
end 

function terminal_con(x, u, w) 
    [
     (x - xT)[collect([(1)..., (6)...])]; # goal 
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT];

# ## rollout
x1 = [q0; q1]
ū = [t < 5 ? [1.0; 0.0; 1.5; 0.0] : t < 10 ? [1.0; 0.0; 1.5; 0.0] : t < 20 ? [0.1; 0.0; 0.1; 0.0] : [0.1; 0.1; 0.1; 0.1] for t = 1:T-1]

w = [[(0.000+ 0.00 * rand()) * rand([-1, 1])] for t = 1:T] # 0.005 + 0.01 baseline
# rollout with list of ilqr dynamics model and initial state, control input with disturbance
x̄, gamma_hist = iLQR.rollout(ilqr_dyns, x1, ū, w)


q̄ = state_to_configuration(x̄)
# visualize!(vis, lineplanarpush, q̄, Δt=h);
# ## solver

solver = iLQR.solver(ilqr_dyns, obj, cons, 
	opts=iLQR.Options(
		linesearch = :armijo,
		α_min=1.0e-5,
		obj_tol=1.0e-3,
		grad_tol=1.0e-3,
		max_iter=10,
		max_al_iter=20,
		con_tol=0.005,
		ρ_init=1.0, 
		ρ_scale=10.0, 
		verbose=false))
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄);

# disturbance
solver.m_data.w = w
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver)

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost

# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
gamma_sol = iLQR.get_contact_force(solver)
q_sol = state_to_configuration(x_sol)

# ## torque distrubance
# Random.seed!(1234)
# uw = [[(uw_values[test_num_w]+ 0.01 * rand()) * rand([-1, 1])] for t = 1:T] # 0.005 + 0.01 baseline

# x_temp, gamma_actual = iLQR.rollout(ilqr_dyns, x1, u_sol, w)
# x_dist, gamma_hist_dist = iLQR.rollout(ilqr_dyns, x1, u_sol, uw)
# q_dist = state_to_configuration(x_dist)

# using Plots
# θ_sol = [q_sol[i][1] for i in 1:T]
# θ_dist = [q_dist[i][1] for i in 1:T]
# time = collect(0:h:(T-1)*h)

# # Plot for θ_sol with θ_goal
# θ_goal_line = fill(θ_goal, length(time))  # Create constant array for θ_goal
# plot(time, θ_sol, label="actual_θ", linewidth=2, color=:green)
# plot!(time, θ_dist, label="dist_θ", linewidth=2, color=:red)
# plot!(time, θ_goal_line, label="goal_θ", linewidth=2, color=:black, linestyle=:dash)
# title!("[line] θ with dist (θ_goal=$(rot_goal[test_number]), uw=$(uw_values[test_num_w]))")
# xlabel!("Time (s)")
# ylabel!("Rotation (rad)")
# savefig("data/line_θ_goal_$(rot_goal[test_number])_$(uw_values[test_num_w]).png")

# # Plot for gamma_sol vs gamma_hist_dist
# gamma_sol_vals = [gamma_actual[i][1] for i in 1:T-1]  # Extract first component of gamma_sol
# gamma_sol_vals2 = [gamma_actual[i][2] for i in 1:T-1]  # Extract first component of gamma_sol
# gamma_hist_dist_vals = [gamma_hist_dist[i][1] for i in 1:T-1]  # Extract first component of gamma_hist_dist
# gamma_hist_dist_vals2 = [gamma_hist_dist[i][2] for i in 1:T-1]  # Extract first component of gamma_hist_dist
# time_controls = collect(0:h:(T-2)*h)  # Time vector for T-1 steps

# # plot(time_controls, gamma_sol_vals, label="γ_actual_1", linewidth=2, color=:green, linestyle=:dash)
# # plot!(time_controls, gamma_sol_vals2, label="γ_actual_2", linewidth=2, color=:green) #, linestyle=:dash)
# # plot!(time_controls, gamma_hist_dist_vals, label="γ_dist_1", linewidth=2, color=:red, linestyle=:dash)
# # plot!(time_controls, gamma_hist_dist_vals2, label="γ_dist_2", linewidth=2, color=:red) #, linestyle=:dash)
# plot(time_controls, gamma_sol_vals.+gamma_hist_dist_vals, label="γ_actual", linewidth=2, color=:green)
# plot!(time_controls, gamma_hist_dist_vals.+gamma_hist_dist_vals2, label="γ_dist", linewidth=2, color=:red)
# title!("[line] Contact Force (θ_goal=$(rot_goal[test_number]), uw=$(uw_values[test_num_w]))")
# xlabel!("Time (s)")
# ylabel!("Contact Force")
# savefig("data/line_contact_force_$(rot_goal[test_number])_$(uw_values[test_num_w]).png")

# # # Plot for u_sol
# u1_vals = [u_sol[i][1] for i in 1:T-1]  # First component of u_sol
# u2_vals = [u_sol[i][2] for i in 1:T-1]  # Second component of u_sol
# u3_vals = [u_sol[i][3] for i in 1:T-1]  # Third component of u_sol
# u4_vals = [u_sol[i][4] for i in 1:T-1]  # Forth component of u_sol
# time_controls = collect(0:h:(T-2)*h)  # Time vector for T-1 steps

# plot(time_controls, u1_vals, label="u_x1", linewidth=2, color=:blue)
# plot!(time_controls, u2_vals, label="u_y1", linewidth=2, color=:green)
# plot!(time_controls, u3_vals, label="u_x2", linewidth=2, color=:blue, linestyle=:dash)
# plot!(time_controls, u4_vals, label="u_y2", linewidth=2, color=:green, linestyle=:dash)
# title!("[line] Control Inputs (θ_goal=$(rot_goal[test_number]))")
# xlabel!("Time (s)")
# ylabel!("Control Input")
# savefig("data/line_control_inputs_$(rot_goal[test_number]).png")

# ## visualization 
# vis = Visualizer() 
# render(vis);
# visualize_with_trail!(vis, lineplanarpush, q_sol, Δt=h);

# vis2 = Visualizer() 
# render(vis2);
# visualize_with_trail!(vis2, lineplanarpush, q_dist, Δt=h);

vis = Visualizer() 
render(vis);
visualize_with_trail!(vis, lineplanarpush, q_sol[1:4], Δt=h);

vis2 = Visualizer() 
render(vis2);
visualize_with_trail!(vis2, lineplanarpush, q_sol[4:7], Δt=h);

vis3 = Visualizer() 
render(vis3);
visualize_with_trail!(vis3, lineplanarpush, q_sol[7:10], Δt=h);

vis4 = Visualizer() 
render(vis4);
visualize_with_trail!(vis4, lineplanarpush, q_sol[10:13], Δt=h);

vis5 = Visualizer() 
render(vis5);
visualize_with_trail!(vis5, lineplanarpush, q_sol[13:16], Δt=h);

vis6 = Visualizer() 
render(vis6);
visualize_with_trail!(vis6, lineplanarpush, q_sol[16:19], Δt=h);

vis7 = Visualizer() 
render(vis7);
visualize_with_trail!(vis7, lineplanarpush, q_sol[19:22], Δt=h);

vis8 = Visualizer() 
render(vis8);
visualize_with_trail!(vis8, lineplanarpush, q_sol[22:25], Δt=h);


# vis5 = Visualizer() 
# render(vis5);
# visualize!(vis5, lineplanarpush, q_sol, Δt=h);

# using CSV
# using DataFrames
# function save_to_csv(q_sol, u_sol, q_dist, gamma_sol_vals, gamma_hist_dist_vals, T;
# 	filename_q="data/line_q_actual_$(rot_goal[test_number]).csv",
# 	filename_u="data/line_u_actual_$(rot_goal[test_number]).csv",
# 	filename_q_dist="data/line_q_dist_$(rot_goal[test_number])_$(uw_values[test_num_w]).csv",
# 	filename_gamma_sol="data/line_gamma_sol_$(rot_goal[test_number])_$(uw_values[test_num_w]).csv",
# 	filename_gamma_dist_sol="data/line_gamma_dist_sol_$(rot_goal[test_number])_$(uw_values[test_num_w]).csv")

#     # q_sol 저장
#     nq = length(q_sol[1]) 
#     q_data = DataFrame([getindex.(q_sol, i) for i in 1:nq], ["q_$i" for i in 1:nq])
#     CSV.write(filename_q, q_data)

# 	# q_sol 저장
#     nq = length(q_dist[1]) 
#     q_dist_data = DataFrame([getindex.(q_dist, i) for i in 1:nq], ["q_dist_$i" for i in 1:nq])
#     CSV.write(filename_q_dist, q_dist_data)

#     # u_sol 저장
#     nu = length(u_sol[1])
#     u_data = DataFrame([getindex.(u_sol, i) for i in 1:nu], ["u_$i" for i in 1:nu])
#     CSV.write(filename_u, u_data)

#     # gamma_sol 저장 (그냥 벡터라 바로 DF로)
#     gamma_sol_data = DataFrame(gamma_sol=(gamma_sol_vals))
#     CSV.write(filename_gamma_sol, gamma_sol_data)

#     # gamma_hist_dist 저장
#     gamma_dist_sol_data = DataFrame(gamma_sol_dist=(gamma_hist_dist_vals))
#     CSV.write(filename_gamma_dist_sol, gamma_dist_sol_data)
# end

# gamma = gamma_sol_vals .+ gamma_sol_vals2
# gamma_dist = gamma_hist_dist_vals .+ gamma_hist_dist_vals2
# save_to_csv(q_sol, u_sol, q_dist, gamma, gamma_dist, T)
# using CSV
# using DataFrames
# function save_to_csv(q_sol, u_sol, T, filename_q="data/qdist_without.csv", filename_u="data/udist_without.csv")
#     # q_sol 저장
#     nq = length(q_sol[1]) 
#     q_data = DataFrame([getindex.(q_sol, i) for i in 1:nq], ["q_$i" for i in 1:nq])
#     CSV.write(filename_q, q_data)

#     # u_sol 저장
#     nu = length(u_sol[1])
#     u_data = DataFrame([getindex.(u_sol, i) for i in 1:nu], ["u_$i" for i in 1:nu])
#     CSV.write(filename_u, u_data)
# end

# # save_to_csv(q_nom, u_nom, T)
# save_to_csv(q_sol, u_sol, T)