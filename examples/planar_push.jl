####
using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

function module_dir()
    @__DIR__
end

# ## visualization 
vis = Visualizer() 
render(vis);

# ## mode
# MODE = :translate
MODE = :rotate 

# ## gradient bundle
GB = false 

# ## state-space model
h = 0.1
T = 26
nc = 5
nc_impact = 1
nw = 0
im_dyn = ImplicitDynamics(planarpush, h, eval(r_pp_func), eval(rz_pp_func), eval(rθ_pp_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-2, nc=1, nb=9, info=(GB ? GradientBundle(planarpush, N=50, ϵ=1.0e-4) : nothing)) 

nx = 2 * planarpush.nq
nu = planarpush.nu 
# ## iLQR model
ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
	(dx, x, u, w) -> GB ? fx_gb(dx, im_dyn, x, u, w) : fx(dx, im_dyn, x, u, w), 
	(du, x, u, w) -> GB ? fu_gb(du, im_dyn, x, u, w) : fu(du, im_dyn, x, u, w), 
	(gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dyn, x, u, w),
	nx, nx, nu, nw, nc, nc_impact) 

model = [ilqr_dyn for t = 1:T-1];
# ## initial conditions and goal
r_dim = 0.1
if MODE == :translate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	x_goal = 0.0
	y_goal = 0.0
	θ_goal = 0.0 * π
	# qT = [x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal - r_dim]
	qT = [x_goal, y_goal, θ_goal, -r_dim - 1.0e-8, -r_dim]

	# [0.0, 0.0, 0.0, -r_dim - 1.0e-8, 0.0]
	#[x_goal, y_goal, θ_goal, x_goal - r_dim, y_goal - r_dim]
	xT = [qT; qT]
elseif MODE == :rotate 
	q0 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	q1 = [0.0, 0.0, 0.0, -r_dim - 1.0e-8, -0.01]
	x1 = [q1; q1]
	x_goal = 0.5
	y_goal = 0.5
	θ_goal = 1.5707963267949#0.5 * π
	qT = [x_goal, y_goal, θ_goal, x_goal-r_dim, y_goal-r_dim]
	xT = [qT; qT]
end

# ## objective
function objt(x, u, w)
	J = 0.0 

	q1 = x[1:planarpush.nq] 
	q2 = x[planarpush.nq .+ (1:planarpush.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT) 
	J += 0.5 * (MODE == :translate ? 1.0e-1 : 1.0e-2) * transpose(u) * u

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:planarpush.nq] 
	q2 = x[planarpush.nq .+ (1:planarpush.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT) 

	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

# ## constraints
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
     (x - xT)[collect([(1:3)..., (6:8)...])]; # goal 
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for t = 1:T-1]..., conT];

# ## rollout
x1 = [q0; q1]
w = [[(0.000+ 0.00 * rand()) * rand([-1, 1])] for t = 1:T] # 0.005 + 0.01 baseline
ū = MODE == :translate ? [t < 5 ? [1.0; 0.0] : [0.0; 0.0] for t = 1:T-1] : [t < 5 ? [1.0; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
x̄, gamma_hist, cv_hist, ip_z_hist, ip_θ_hist = iLQR.rollout(model, x1, ū, w) #gamma_hist
# for i=1:T
#     println(i, " : ", x̄[i])
# end
q̄ = state_to_configuration(x̄)
visualize!(vis, planarpush, q̄, Δt=h);

# ## solver
solver = iLQR.solver(model, obj, cons, 
	opts=iLQR.Options(
		linesearch = :armijo,
		α_min=1.0e-5,
		obj_tol=1.0e-3,
		grad_tol=1.0e-3,
		max_iter=10,
		max_al_iter=10,
		con_tol=0.005,
		ρ_init=1.0, 
		ρ_scale=10.0, 
		verbose=false))
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄);

# ## solve
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver);

# @show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
# @show solver.s_data.iter[1]
# @show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
# @show solver.s_data.obj[1] # augmented Lagrangian cost
		
# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
q_sol = state_to_configuration(x_sol)
x_rollout_sol, gamma_hist, b_hist, ip_z_hist, ip_θ_hist = iLQR.rollout(model, x1, u_sol, w)
@show u_sol
@show x_rollout_sol
q_rollout_sol = state_to_configuration(x_rollout_sol)
visualize!(vis, planarpush, q_rollout_sol, Δt=h);

# # JLD2 파일 저장
gait_path = joinpath(module_dir(), "pusher_ref_traj.jld2")
save_trajectory(gait_path, x_rollout_sol, u_sol, gamma_hist, b_hist, ip_z_hist, ip_θ_hist, w, model, h, T)

# 저장된 데이터 로드 (확인용)
# ref_traj = deepcopy(get_trajectory(planarpush, env, gait_path, load_type=:joint_traj))

q_sol = state_to_configuration(x_sol)
visualize!(vis, planarpush, q_sol, Δt=h);

# ## benchmark 
# solver.options.verbose = false
# @benchmark iLQR.solve!($solver, x̄, ū) setup=(x̄=deepcopy(x̄), ū=deepcopy(ū));
