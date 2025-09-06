using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

# ## mode
# MODE = :translate
MODE = :rotate 

# ## gradient bundle
GB = false 

h = 0.1
T = 26
num_w = 1
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
	nx, nx, nu, num_w) 

# generate ilqr dynamics for each trajectory index
ilqr_dyns = [ilqr_dyn for t = 1:T-1];
# ## initial conditions and goal
r_dim = 0.1
if MODE == :translate 
	q0 = [0.0, -r_dim - 1.0e-8, 0.0]
	q1 = [0.0, -r_dim - 1.0e-8, 0.0]
	x_goal = 1.0
	y_goal = 0.0
	θ_goal = 0.0 * π
	qT = [θ_goal,  - r_dim,  - r_dim]
	xT = [qT; qT]
elseif MODE == :rotate 
	q0 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025]
	q1 = [0.0, -r_dim - 1.0e-8, 0.025, -r_dim - 1.0e-8, -0.025]
	x1 = [q1; q1]
	θ_goal = 0.175*3
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
	J += 0.5 * (MODE == :translate ? 1.0e-1 : 1.0e-2) * transpose(u) * u
	ϕ = ϕ_func(lineplanarpush, q2)[1]  # SDF 값 (접촉 시 0)
	J += 0.5 * (ϕ+1e-8)^2
	ϕ = ϕ_func(lineplanarpush, q2)[2]  # SDF 값 (접촉 시 0)
	J += 0.5 * (ϕ+1e-8)^2

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
	J += 0.5 * (ϕ+1e-8)^2
	ϕ = ϕ_func(lineplanarpush, q2)[2]  # SDF 값 (접촉 시 0)
	J += 0.5 * (ϕ+1e-8)^2

	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

# ## constraints
ul = [-10.0; -10.0; -10.0; -10.0]
uu = [10.0; 10.0; 10.0; 10.0]

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
ū = [t < 5 ? [0.5; 0.0; 2.5; 0.0] : t < 10 ? [1.0; 0.0; 2.5; 0.0] : t < 20 ? [0.1; 0.0; 0.1; 0.0] : [0.1; 0.1; 0.1; 0.1] for t = 1:T-1]

w = [[(0.000+ 0.00 * rand()) * rand([-1, 1])] for t = 1:T] # 0.005 + 0.01 baseline
# rollout with list of ilqr dynamics model and initial state, control input with disturbance
x̄, gamma_hist = iLQR.rollout(ilqr_dyns, x1, ū, w)

for i=1:T
    println(i, " : ", x̄[i])
end
for i=1:T-1
    println(i, " : ", gamma_hist[i])
end
for i=1:T-1
    println(i, " : ", ū[i])
end
# ## visualization 
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

using Random

# 시드 고정
Random.seed!(1234) # 원하는 시드 값(예: 1234)을 설정
uw = [[(0.005+ 0.05 * rand()) * rand([-1, 1])] for t = 1:T] # 0.005 + 0.01 baseline
# for i=1:T-1
# 	u_sol[i][1] = u_sol[i][1] + uw[i][1]
# 	u_sol[i][2] = u_sol[i][2] + uw[i][1]
# 	u_sol[i][3] = u_sol[i][3] + uw[i][1]
# 	u_sol[i][4] = u_sol[i][4] + uw[i][1]
# end
x_dist, gamma_hist_dist = iLQR.rollout(ilqr_dyns, x1, u_sol, uw)
q_dist = state_to_configuration(x_dist)
# # Compare
# for i = 1:T
#     println("Nominal q[$i]: ", q_nom[i][1:5], " | Disturbed q[$i]: ", q_dist[i][1:5])
# end

# # θ_nom = [q_nom[i][1] for i in 1:T]
# θ_sol = [q_sol[i][1] for i in 1:T]
# h = 0.1
# time = collect(0:h:(T-1)*h)
# using Plots
# plot(time, θ_sol, label="Disturbed θ", linewidth=2, color=:red)
# title!("object θ with disturbance")
# # title!("object θ without disturbance")
# xlabel!("Time (s)")
# ylabel!("Rotation (rad)")
# savefig("theta_comparison.png")

# ## solution
for i=1:T-1
	println("u_sol :", u_sol[i])
end

for i=1:T-1
	println("gamma_sol :", gamma_sol[i])
end

for i=1:T
	println("qsol :", q_sol[i])
	println("q_dist :", q_dist[i])
end

for i=1:T-1
	println("disturbance :", uw[i])
end

# # ## visualization 
vis = Visualizer() 
render(vis);
visualize!(vis, lineplanarpush, q_sol, Δt=h);

vis2 = Visualizer() 
render(vis2);
visualize!(vis2, lineplanarpush, q_dist, Δt=h);


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