using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
using LinearAlgebra
using Random

# ## mode
# MODE = :translate
MODE = :rotate 

# ## gradient bundle
GB = false 

# ## state-space model
h = 0.1
T = 26

im_dyn = ImplicitDynamics(lineplanarpush, h, eval(r_lpp_func), eval(rz_lpp_func), eval(rθ_lpp_func); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-2, nc=2, nb=10, d=1, info=(GB ? GradientBundle(lineplanarpush, N=50, ϵ=1.0e-4) : nothing)) 

nx = 2 * lineplanarpush.nq
nu = lineplanarpush.nu 

# ## iLQR model
ilqr_dyn = iLQR.Dynamics((d, x, u, w) -> f(d, im_dyn, x, u, w), 
	(dx, x, u, w) -> GB ? fx_gb(dx, im_dyn, x, u, w) : fx(dx, im_dyn, x, u, w), 
	(du, x, u, w) -> GB ? fu_gb(du, im_dyn, x, u, w) : fu(du, im_dyn, x, u, w), 
	nx, nx, nu) 

model = [ilqr_dyn for t = 1:T-1];
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
	θ_goal = 0.5
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

	return J
end

function objT(x, u, w)
	J = 0.0 
	
	q1 = x[1:lineplanarpush.nq] 
	q2 = x[lineplanarpush.nq .+ (1:lineplanarpush.nq)] 
	v1 = (q2 - q1) ./ h

	J += 0.5 * transpose(v1) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1]) * v1 
	J += 0.5 * transpose(x - xT) * Diagonal([1.0, 0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 0.1]) * (x - xT) 
	return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for t = 1:T-1]..., cT];

# ## constraints
ul = [0.0; -10.0; 0.0; -10.0]
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
ū = MODE == :translate ? [t < 5 ? [0.0; 0.0] : [0.0; 0.0] for t = 1:T-1] : [t < 5 ? [0.5; 0.0; 1.5; 0.0] : t < 10 ? [1.0; 0.0; 2.5; 0.0] : [0.1; 0.1; 0.1; 0.1] for t = 1:T-1]
x̄ = iLQR.rollout(model, x1, ū)
for i=1:T
    println(i, " : ", x̄[i])
end
for i=1:T-1
    println(i, " : ", ū[i])
end
q̄ = state_to_configuration(x̄)
# visualize!(vis, lineplanarpush, q̄, Δt=h);

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
solver.m_data.w = [[0] for t = 1:T]
# ## solve
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver);

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0)), Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost
		
# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
q_sol = state_to_configuration(x_sol)
for i=1:T-1
	println("u_sol :", u_sol[i])
end

for i=1:T
	println("qsol :", q_sol[i])
end

# ## visualization 
vis = Visualizer() 
render(vis);

visualize!(vis, lineplanarpush, q_sol, Δt=h);


using CSV
using DataFrames
function save_to_csv(q_sol, u_sol, T, filename_q="data/qsol.csv", filename_u="data/usol.csv")
    # q_sol 저장
    nq = length(q_sol[1])  # 상태 벡터의 차원
    q_data = DataFrame([getindex.(q_sol, i) for i in 1:nq], ["q_$i" for i in 1:nq])
    CSV.write(filename_q, q_data)

    # u_sol 저장
    nu = length(u_sol[1])  # 제어 입력 벡터의 차원
    u_data = DataFrame([getindex.(u_sol, i) for i in 1:nu], ["u_$i" for i in 1:nu])
    CSV.write(filename_u, u_data)
end

save_to_csv(q_sol, u_sol, T)