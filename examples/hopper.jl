using Plots
using Random
Random.seed!(1)

include_implicit_dynamics()
include_ddp()
include(joinpath(pwd(), "examples/implicit_dynamics/models/hopper_2D/model.jl"))

# Time
T = 21
h = 0.05

# Implicit dynamics
data = dynamics_data(model, h, T,
        r_func, rz_func, rθ_func, rz_array, rθ_array;
        idx_ineq = idx_ineq,
		z_subset_init = z_subset_init,
        diff_idx = 4,
        opts =  InteriorPointOptions{Float64}(
						r_tol = 1.0e-8,
						κ_tol = 1.0e-4,
						κ_init = 1.0,
						diff_sol = false))
# implicit model cache
fdx1 = zeros(4 * model.dim.q, 2 * model.dim.q)
fdxt = [zeros(model.dim.q, model.dim.q) I zeros(model.dim.q, 2 * model.dim.q);
        zeros(model.dim.q, 4 * model.dim.q);
        zeros(2 * model.dim.q, 2 * model.dim.q) I]
fdu1 = [zeros(model.dim.q, model.dim.u + model.dim.q) I;
        zeros(model.dim.q, model.dim.u + 2 * model.dim.q);
        zeros(2 * model.dim.q, model.dim.u) I]
fdut = [zeros(model.dim.q, model.dim.u);
        zeros(model.dim.q, model.dim.u);
        zeros(2 * model.dim.q, model.dim.u)]

cache = Dict(:fdx1 => fdx1,
             :fdxt => fdxt,
             :fdu1 => fdu1,
             :fdut => fdut)

model_implicit = ImplicitDynamics{Midpoint, FixedTime}(2 * model.dim.q, model.dim.u, 0, data, cache = cache)

# Parameter optimization discrete-time dynamics
n = [t == 1 ? model_implicit.n : 2 * model_implicit.n for t = 1:T]
m = [t == 1 ? model_implicit.m + model_implicit.n : model_implicit.m for t = 1:T-1]

function fd(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	np = model.n
	nq = model.dynamics.m.dim.q
	nu = model.dynamics.m.dim.u

	if t == 1
		θ = view(u, nu .+ (1:np))
		q0 = view(θ, 1:nq)
		q1 = view(θ, nq .+ (1:nq))
	else
		θ = view(x, 2 * nq .+ (1:np))
		q0 = view(x, 1:nq)
		q1 = view(x, nq .+ (1:nq))
	end

	u1 = view(u, 1:nu)

	q2 = f(model.dynamics, q0, q1, u1, t)

	return [q1; q2; θ]
end

function fdx(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	np = model.n
	nq = model.dynamics.m.dim.q
	nu = model.dynamics.m.dim.u

	if t == 1
		θ = view(u, nu .+ (1:np))
		q0 = view(θ, 1:nq)
		q1 = view(θ, nq .+ (1:nq))
	else
		θ = view(x, 2 * nq .+ (1:np))
		q0 = view(x, 1:nq)
		q1 = view(x, nq .+ (1:nq))
	end

	u1 = view(u, 1:nu)

	dq2dx1 = fx1(model.dynamics, q0, q1, u1, t)

	if t == 1
        return model.cache[:fdx1]
	else
        model.cache[:fdxt][nq .+ (1:nq), 1:2nq] = dq2dx1
        return model.cache[:fdxt]
	end
end

function fdu(model::ImplicitDynamics{Midpoint, FixedTime}, x, u, w, h, t)
	np = model.n
	nq = model.dynamics.m.dim.q
	nu = model.dynamics.m.dim.u

	if t == 1
		θ = view(u, nu .+ (1:np))
		q0 = view(θ, 1:nq)
		q1 = view(θ, nq .+ (1:nq))
	else
		θ = view(x, 2 * nq .+ (1:np))
		q0 = view(x, 1:nq)
		q1 = view(x, nq .+ (1:nq))
	end

	u1 = view(u, 1:nu)

	dq2dx1 = fx1(model.dynamics, q0, q1, u1, t)
	dq2du1 = fu1(model.dynamics, q0, q1, u1, t)

	if t == 1
        model.cache[:fdu1][nq .+ (1:nq), 1:nu] = dq2du1
        model.cache[:fdu1][nq .+ (1:nq), nu .+ (1:2nq)] = dq2dx1
        return model.cache[:fdu1]
	else
        model.cache[:fdut][nq .+ (1:nq), 1:nu] = dq2du1
        return model.cache[:fdut]
	end
end

# Initial conditions, controls, disturbances
q1 = [0.0; 0.5; 0.0; 0.5]
qM = [0.5; 0.5; 0.0; 0.5]
qT = [1.0; 0.5; 0.0; 0.5]
q_ref = [0.5; 0.75; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

ū_stand = [t == 1 ? [0.0; model_implicit.dynamics.m.g * (model_implicit.dynamics.m.mb + model_implicit.dynamics.m.ml) * 0.5 * h; x1] : [0.0; model_implicit.dynamics.m.g * (model_implicit.dynamics.m.mb + model_implicit.dynamics.m.ml) * 0.5 * h] for t = 1:T-1]
w = [zeros(model_implicit.d) for t = 1:T-1]

# Rollout
x̄ = rollout(model_implicit, x1, ū_stand, w, h, T)

# Objective

# gait 1
# Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
# 	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(model_implicit.n)])
# 	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(model_implicit.n)])) for t = 1:T]
# q = [t == 1 ? -2.0 * Q[t] * (t < 6 ? x_ref : x_ref) : -2.0 * Q[t] * [(t < 6 ? x_ref : x_ref); zeros(model_implicit.n)] for t = 1:T]
# R = [t == 1 ? Diagonal([1.0e-1 * ones(model_implicit.m); 1.0e-1 * ones(nq); 1.0e-5 * ones(nq)]) : Diagonal(1.0e-1 * ones(model_implicit.m)) for t = 1:T-1]
# r = [t == 1 ? [zeros(model_implicit.m); -2.0 * R[t][1:nq, 1:nq] * x1[1:nq]; zeros(nq)] : zeros(model_implicit.m) for t = 1:T-1]

# gait 2
# Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
# 	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(model_implicit.n)])
# 	: 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(model_implicit.n)])) for t = 1:T]
# q = [t == 1 ? -2.0 * Q[t] * (t < 6 ? x_ref : x_ref) : -2.0 * Q[t] * [(t < 6 ? x_ref : x_ref); zeros(model_implicit.n)] for t = 1:T]
# R = [t == 1 ? Diagonal([1.0 * ones(model_implicit.m); 1.0e-1 * ones(nq); 1.0e-5 * ones(nq)]) : Diagonal(1.0 * ones(model_implicit.m)) for t = 1:T-1]
# r = [t == 1 ? [zeros(model_implicit.m); -2.0 * R[t][1:nq, 1:nq] * x1[1:nq]; zeros(nq)] : zeros(model_implicit.m) for t = 1:T-1]

# gait 3
Q = [(t == 1 ? 1.0 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0])
	: t == T ? Diagonal([1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; 1.0; zeros(model_implicit.n)])
	: 0.1 * Diagonal([1.0; 10.0; 1.0; 10.0; 1.0; 10.0; 1.0; 10.0; zeros(model_implicit.n)])) for t = 1:T]
q = [t == 1 ? -2.0 * Q[t] * (t < 6 ? x_ref : x_ref) : -2.0 * Q[t] * [(t < 6 ? x_ref : x_ref); zeros(model_implicit.n)] for t = 1:T]
R = [t == 1 ? Diagonal([1.0e-3 * ones(model_implicit.m); 1.0e-1 * ones(nq); 1.0e-5 * ones(nq)]) : Diagonal(1.0e-3 * ones(model_implicit.m)) for t = 1:T-1]
r = [t == 1 ? [zeros(model_implicit.m); -2.0 * R[t][1:nq, 1:nq] * x1[1:nq]; zeros(nq)] : zeros(model_implicit.m) for t = 1:T-1]

obj = StageCosts([QuadraticCost(Q[t], q[t],
	t < T ? R[t] : nothing, t < T ? r[t] : nothing) for t = 1:T], T)

function g(obj::StageCosts, x, u, t)
	T = obj.T
    if t < T
		Q = obj.cost[t].Q
		q = obj.cost[t].q
	    R = obj.cost[t].R
		r = obj.cost[t].r
        return x' * Q * x + q' * x + u' * R * u + r' * u
    elseif t == T
		Q = obj.cost[T].Q
		q = obj.cost[T].q
        return x' * Q * x + q' * x
    else
        return 0.0
    end
end

# Constraints
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
p = [t < T ? (t == 1 ? 2 * model_implicit.m + nq + 2 * 2 : (t == 6 ? 2 * model_implicit.m + 0 * model_implicit.n : 2 * model_implicit.m)) : (model_implicit.n - 2) + 2 for t = 1:T]
info_1 = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model_implicit.m))
info_t = Dict(:ul => ul, :uu => uu, :inequality => (1:2 * model_implicit.m))
info_T = Dict(:xT => xT, :inequality => (1:2))
con_set = [StageConstraint(p[t], t < T ? (t == 1 ? info_1 : info_t) : info_T) for t = 1:T]

function c!(c, cons::StageConstraints, x, u, t)
	T = cons.T
	p = cons.con[t].p

	if t < T
		ul = cons.con[t].info[:ul]
		uu = cons.con[t].info[:uu]
		u1 = view(u, 1:model_implicit.m)
		c[1:(2 * model_implicit.m)] .= [ul - u1; u1 - uu]

		if t == 1
			c[2 * model_implicit.m .+ (1:nq)] .= view(u, model_implicit.m .+ (1:nq)) - view(x1, 1:nq)
			c[2 * model_implicit.m + nq .+ (1:2)] = kinematics(model_implicit.dynamics.m,
                view(u, model_implicit.m .+ (1:nq))) - kinematics(model_implicit.dynamics.m, view(x1, 1:nq))
			c[2 * model_implicit.m + nq + 2 .+ (1:2)] = kinematics(model_implicit.dynamics.m,
                view(u, model_implicit.m + nq .+ (1:nq))) - kinematics(model_implicit.dynamics.m, view(x1, nq .+ (1:nq)))
		end
	end

	if t == T
		x_travel = 0.5
		θ = view(x, model_implicit.n .+ (1:model_implicit.n))

		c[1] = x_travel - (x[1] - θ[1])
		c[2] = x_travel - (x[nq + 1] - θ[nq + 1])
		c[2 .+ (1:3)] = x[1:nq][collect([2, 3, 4])] - θ[1:nq][collect([2, 3, 4])]
		c[2 + 3 .+ (1:3)] = x[nq .+ (1:nq)][collect([2, 3, 4])] - θ[nq .+ (1:nq)][collect([2, 3, 4])]
	end
end

prob = problem_data(model_implicit, obj, con_set, copy(x̄), copy(ū_stand), w, h, T,
 	n = n, m = m,
	analytical_dynamics_derivatives = true)

# Solve
@time stats = constrained_ddp_solve!(prob,
    verbose = true,
    grad_tol = 1.0e-3,
	max_iter = 100,
    max_al_iter = 10,
	ρ_init = 1.0,
    ρ_scale = 10.0,
	con_tol = 1.0e-3)

@show ilqr_iterations(stats)
@show objective(obj, prob.m_data.x̄, prob.m_data.ū)
@show prob.s_data.c_max

x, u = current_trajectory(prob)
x̄, ū = nominal_trajectory(prob)

J = 0.0
x = x̄
u = ū
for t = 1:T
    if t < T
        J += x[t][1:8]' * Q[t][1:8, 1:8] * x[t][1:8] + q[t][1:8]' * x[t][1:8] + u[t][1:2]' * R[t][1:2, 1:2] * u[t][1:2] + r[t][1:2]' * u[t][1:2]
    elseif t == T
        J += x[t][1:8]' * Q[t][1:8, 1:8] * x[t][1:8] + q[t][1:8]' * x[t][1:8]
    else
        J += 0.0
    end
end
@show J

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1.jld2") x u
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_1.jld2") x u

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2.jld2") x u
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_2.jld2") x u

# @save joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3.jld2") x u
# @load joinpath(pwd(), "examples/implicit_dynamics/examples/trajectories/hopper_gait_3.jld2") x u

q̄ = state_to_configuration(x̄)

# update initial state using optimized parameter
q̄[1] = ū[1][model_implicit.m .+ (1:nq)]
q̄[2] = ū[1][model_implicit.m + nq .+ (1:nq)]

include(joinpath(pwd(), "models/visualize.jl"))
include(joinpath(pwd(), "examples/implicit_dynamics/models/hopper_2D/visuals.jl"))

vis = Visualizer()
render(vis)
visualize!(vis, model, q̄[1:end], Δt = h)
#
#  x̄[T][1:model_implicit.n] - ū[1][model_implicit.m .+ (1:model_implicit.n)]
#
function mirror_gait(q, T; n = 5)
	qm = [deepcopy(q)...]
	um = [deepcopy(u)...]

	stride = zero(qm[1])
	strd = q[T+1][1] - q[2][1]
	@show stride[1] += strd
	@show 0.5 * stride

	for i = 1:n-1
		for t = 1:T-1
			push!(qm, q[t+2] + stride)
			push!(um, u[t])
		end
		stride[1] += strd
	end
	len = qm[end][1]

	# center
	for t = 1:length(qm)
		qm[t][1] -= 0.5 * len
	end

	return qm, um
end

qm, um = mirror_gait(q̄, T)
visualize!(vis, model, qm, Δt = h)
settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, -95.0, -1.0), LinearMap(RotY(0.0 * π) * RotZ(-π / 2.0))))
setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
#
open(vis)
#
# body_points = Vector{Point{3,Float64}}()
# foot_points = Vector{Point{3,Float64}}()
# body_points_opt = Vector{Point{3,Float64}}()
# foot_points_opt = Vector{Point{3,Float64}}()
#
# for q in qm
# 	push!(body_points, Point(q[1], 0.01, q[2] + 0.05))
# 	push!(body_points_opt, Point(q[1], -0.0, q[2] + 0.05))
#
# 	k = kinematics(model, q)
# 	push!(foot_points, Point(k[1], 0.01, k[2] + 0.05))
# 	push!(foot_points_opt, Point(k[1], -0.0, k[2] + 0.05))
# end
#
# line_opt_mat = LineBasicMaterial(color=color=Colors.RGBA(1.0, 0.0, 0.0, 1.0), linewidth=10.0)
# body_line_mat = LineBasicMaterial(color=color=Colors.RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0), linewidth=5.0)
# foot_line_mat = LineBasicMaterial(color=color=Colors.RGBA(51.0 / 255.0, 1.0, 1.0, 1.0), linewidth=5.0)
#
# setobject!(vis[:body_traj], MeshCat.Line(body_points, body_line_mat))
# setobject!(vis[:foot_traj], MeshCat.Line(foot_points, foot_line_mat))
# setobject!(vis[:body_traj_opt], MeshCat.Line(body_points_opt[1:T+1], line_opt_mat))
# setobject!(vis[:foot_traj_opt], MeshCat.Line(foot_points_opt[1:T+1], line_opt_mat))
#
# for i = 1:length(body_points)
#     setobject!(vis["body_line_vertex_$i"], Sphere(Point3f0(0),
#         convert(Float32, 0.005)),
#         MeshPhongMaterial(color = Colors.RGBA(1.0, 153.0 / 255.0, 51.0 / 255.0, 1.0)))
#         settransform!(vis["body_line_vertex_$i"], Translation(body_points[i]))
#     setobject!(vis["foot_line_vertex_$i"], Sphere(Point3f0(0),
#         convert(Float32, 0.005)),
#         MeshPhongMaterial(color = Colors.RGBA(51.0 / 255.0, 1.0, 1.0, 1.0)))
#         settransform!(vis["foot_line_vertex_$i"], Translation(foot_points[i]))
#
#     i > T+1 && continue
#     setobject!(vis["body_line_opt_vertex_$i"], Sphere(Point3f0(0),
#         convert(Float32, 0.0075)),
#         MeshPhongMaterial(color = Colors.RGBA(1, 0, 0, 1.0)))
#         settransform!(vis["body_line_opt_vertex_$i"], Translation(body_points_opt[i]))
#     setobject!(vis["foot_line_opt_vertex_$i"], Sphere(Point3f0(0),
#         convert(Float32, 0.0075)),
#         MeshPhongMaterial(color = Colors.RGBA(1, 0, 0, 1.0)))
#         settransform!(vis["foot_line_opt_vertex_$i"], Translation(foot_points_opt[i]))
# end
