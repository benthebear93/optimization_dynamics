using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR
const RoboDojo = OptimizationDynamics.RoboDojo
using LinearAlgebra
using Random
using JSON3
using OrderedCollections: OrderedDict

# ## visualize 
vis = Visualizer() 
render(vis);

# ## state-space model
T = 40
h = 0.05
hopper = RoboDojo.hopper
HOPPER_REF_TRAJ_FILE = joinpath(@__DIR__, "..", "hopper_ref_traj.json")

struct ParameterOptInfo{T}
	idx_q1::Vector{Int} 
	idx_q2::Vector{Int} 
	idx_u1::Vector{Int}
	idx_uθ::Vector{Int}
	idx_uθ1::Vector{Int} 
	idx_uθ2::Vector{Int}
	idx_xθ::Vector{Int}
	v1::Vector{T}
end

info = ParameterOptInfo(
	collect(1:hopper.nq), 
	collect(hopper.nq .+ (1:hopper.nq)), 
	collect(1:hopper.nu), 
	collect(hopper.nu .+ (1:2 * hopper.nq)),
	collect(hopper.nu .+ (1:hopper.nq)), 
	collect(hopper.nu + hopper.nq .+ (1:hopper.nq)), 
	collect(2 * hopper.nq .+ (1:2 * hopper.nq)),
	zeros(hopper.nq)
)

im_dyn1 = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	n=(2 * hopper.nq), m=(hopper.nu + 2 * hopper.nq), nc=4, nb=2, nc_impact=4, info=info)

im_dynt = ImplicitDynamics(hopper, h, 
	eval(RoboDojo.residual_expr(hopper)), 
	eval(RoboDojo.jacobian_var_expr(hopper)), 
	eval(RoboDojo.jacobian_data_expr(hopper)); 
    r_tol=1.0e-8, κ_eval_tol=1.0e-4, κ_grad_tol=1.0e-3,
	n=4 * hopper.nq, m=hopper.nu, nc=4, nb=2, nc_impact=4, info=info) 

function f1(d, model::ImplicitDynamics, x, u, w)

	θ = @views u[model.info.idx_uθ]
	q1 = @views u[model.info.idx_uθ1]
	q2 = @views u[model.info.idx_uθ2]
	u1 = @views u[model.info.idx_u1] 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.eval_sim.h

	q3 = RoboDojo.step!(model.eval_sim, q2, model.info.v1, u1, 1)

	d[model.info.idx_q1] = q2 
	d[model.info.idx_q2] = q3
	d[model.info.idx_xθ] = θ

	return d
end

function f1x(dx, model::ImplicitDynamics, x, u, w)
	dx .= 0.0
	return dx
end

function f1u(du, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	θ = @views u[model.info.idx_uθ]
	q1 = @views u[model.info.idx_uθ1]
	q2 = @views u[model.info.idx_uθ2]
	u1 = @views u[model.info.idx_u1] 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h

	RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	for i = 1:nq
		du[model.info.idx_q1[i], model.info.idx_uθ[i]] = 1.0 
	end
	du[model.info.idx_q2, model.info.idx_u1] = model.grad_sim.grad.∂q3∂u1[1] 
	du[model.info.idx_q2, model.info.idx_uθ1] = model.grad_sim.grad.∂q3∂q1[1] 
	du[model.info.idx_q2, model.info.idx_uθ2] = model.grad_sim.grad.∂q3∂q2[1] 

	return du
end

function ft(d, model::ImplicitDynamics, x, u, w)

	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.eval_sim.h 

	q3 = RoboDojo.step!(model.eval_sim, q2, model.info.v1, u1, 1)

	d[model.info.idx_q1] = q2 
	d[model.info.idx_q2] = q3
	d[model.info.idx_xθ] = θ

	return d
end

function ftx(dx, model::ImplicitDynamics, x, u, w)
	nq = model.grad_sim.model.nq

	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h 

	q3 = RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	for i = 1:nq
		dx[model.info.idx_q1[i], model.info.idx_q2[i]] = 1.0 
	end
	dx[model.info.idx_q2, model.info.idx_q1] = model.grad_sim.grad.∂q3∂q1[1] 
	dx[model.info.idx_q2, model.info.idx_q2] = model.grad_sim.grad.∂q3∂q2[1] 
	for i in model.info.idx_xθ 
		dx[i, i] = 1.0 
	end

	return dx
end
	
function ftu(du, model::ImplicitDynamics, x, u, w)
	θ = @views x[model.info.idx_xθ] 
	q1 = @views x[model.info.idx_q1]
	q2 = @views x[model.info.idx_q2] 
	u1 = u 

	model.info.v1 .= q2 
	model.info.v1 .-= q1 
	model.info.v1 ./= model.grad_sim.h 

	q3 = RoboDojo.step!(model.grad_sim, q2, model.info.v1, u1, 1)

	du[model.info.idx_q2, model.info.idx_u1] = model.grad_sim.grad.∂q3∂u1[1]

	return du
end

# ## iLQR model
nw = hopper.nw
nc = 4
nc_impact = 4

ilqr_dyn1 = iLQR.Dynamics((d, x, u, w) -> f1(d, im_dyn1, x, u, w), 
					(dx, x, u, w) -> f1x(dx, im_dyn1, x, u, w), 
					(du, x, u, w) -> f1u(du, im_dyn1, x, u, w), 
                    (gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dyn1, x, u, w),
					4 * hopper.nq, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, nw, nc, nc_impact)  

ilqr_dynt = iLQR.Dynamics((d, x, u, w) -> ft(d, im_dynt, x, u, w), 
	(dx, x, u, w) -> ftx(dx, im_dynt, x, u, w), 
	(du, x, u, w) -> ftu(du, im_dynt, x, u, w), 
    (gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dynt, x, u, w),
	4 * hopper.nq, 4 * hopper.nq, hopper.nu, nw, nc, nc_impact)  

model = [ilqr_dyn1, [ilqr_dynt for t = 2:T-1]...];

# ## initial conditions
q1 = [0.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
qM = [0.5; 0.5 + hopper.foot_radius; 0.0; 0.5]
qT = [1.0; 0.5 + hopper.foot_radius; 0.0; 0.5]
q_ref = [0.5; 0.75 + hopper.foot_radius; 0.0; 0.25]

x1 = [q1; q1]
xM = [qM; qM]
xT = [qT; qT]
x_ref = [q_ref; q_ref]

# Time-varying forward-hopping reference.
const X_TRAVEL_GOAL = 3.0
const HOP_CYCLES = 6

function hopper_q_ref(t::Int, T::Int, hopper, q_start)
    α = (t - 1) / (T - 1)
    phase = 2.0 * pi * HOP_CYCLES * α
    x = q_start[1] + X_TRAVEL_GOAL * α
    z = (0.5 + hopper.foot_radius) + 0.20 * max(0.0, sin(phase))
    θb = 0.0
    r = clamp(0.5 - 0.12 * sin(phase), hopper.leg_len_min + 0.05, hopper.leg_len_max - 0.05)
    return [x; z; θb; r]
end

q_ref_traj = [hopper_q_ref(t, T, hopper, q1) for t = 1:T]
x_ref_traj = [[q; q] for q in q_ref_traj]

# For long-travel hopping, do not penalize global x translation in stage costs.
const W_STAGE_STATE = Diagonal([0.0; 10.0; 1.0; 10.0; 0.0; 10.0; 1.0; 10.0])
const W_TERM_STATE = Diagonal([0.0; 1.0; 1.0; 1.0; 0.0; 1.0; 1.0; 1.0])

# ## objective

GAIT = 3
## GAIT = 2 
## GAIT = 3

if GAIT == 1 
	r_cost = 1.0e-1 
	q_cost = 1.0e-1
elseif GAIT == 2 
	r_cost = 1.0
	q_cost = 1.0
elseif GAIT == 3 
	r_cost = 1.0e-3
	q_cost = 1.0e-1
end

function obj1(x, u, w, x_ref_local)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref_local) * W_STAGE_STATE * (x - x_ref_local)
	J += 0.5 * transpose(u) * Diagonal([r_cost * ones(hopper.nu); 1.0e-1 * ones(hopper.nq); 1.0e-5 * ones(hopper.nq)]) * u
	return J
end

function objt(x, u, w, x_ref_local)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref_local) * q_cost * Diagonal([diag(W_STAGE_STATE); zeros(2 * hopper.nq)]) * (x - x_ref_local)
    v1 = (x[hopper.nq .+ (1:hopper.nq)] - x[1:hopper.nq]) ./ h
    J += 0.5 * transpose(v1) * Diagonal([25.0; 3.0; 1.0; 1.0]) * v1
	J += 0.5 * transpose(u) * Diagonal(r_cost * ones(hopper.nu)) * u
	return J
end

function objT(x, u, w, x_ref_local)
	J = 0.0 
	J += 0.5 * transpose(x - x_ref_local) * Diagonal([diag(W_TERM_STATE); zeros(2 * hopper.nq)]) * (x - x_ref_local)
    v1 = (x[hopper.nq .+ (1:hopper.nq)] - x[1:hopper.nq]) ./ h
    J += 0.5 * transpose(v1) * Diagonal([60.0; 8.0; 2.0; 2.0]) * v1
	return J
end

c1 = iLQR.Cost((x, u, w) -> obj1(x, u, w, x_ref_traj[1]), 2 * hopper.nq, hopper.nu + 2 * hopper.nq)
cts = [iLQR.Cost((x, u, w) -> objt(x, u, w, [x_ref_traj[t]; zeros(2 * hopper.nq)]), 4 * hopper.nq, hopper.nu) for t = 2:T-1]
cT = iLQR.Cost((x, u, w) -> objT(x, u, w, [x_ref_traj[T]; zeros(2 * hopper.nq)]), 4 * hopper.nq, 0)
obj = [c1, cts..., cT];

function write_hopper_ref_traj_json(
    out_file::String,
    q_sol,
    u_sol,
    h::Float64,
    nq::Int,
    nu::Int,
)
    H = length(u_sol)
    q = [Vector{Float64}(q_sol[t]) for t in 1:H]
    u = [Vector{Float64}(u_sol[t]) for t in 1:H]
    w = [Float64[] for _ = 1:H]
    gamma = [Float64[] for _ = 1:H]
    b = [Float64[] for _ = 1:H]
    z = [Float64[] for _ = 1:H]
    theta = [Float64[] for _ = 1:H]

    nw = 0
    nc = 0
    nb = 0
    nz = 0
    ntheta = 0

    iq0 = collect(1:nq)
    iq1 = collect(nq .+ (1:nq))
    iu1 = collect(2 * nq .+ (1:nu))
    iw1 = Int[]
    iq2 = collect(1:nq)
    igamma1 = Int[]
    ib1 = Int[]

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

# ## constraints
ul = [-10.0; -10.0]
uu = [10.0; 10.0]
 
function stage1_con(x, u, w) 
    vx_stage_max = 1.0
    vx = (x[hopper.nq + 1] - x[1]) / h
    [
    ul - u[1:hopper.nu]; # control limit (lower)
    u[1:hopper.nu] - uu; # control limit (upper)
    vx - vx_stage_max;
    -vx - vx_stage_max;

	u[hopper.nu .+ (1:hopper.nq)] - x1[1:hopper.nq];

	RoboDojo.kinematics_foot(hopper, u[hopper.nu .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[1:hopper.nq]);
	RoboDojo.kinematics_foot(hopper, u[hopper.nu + hopper.nq .+ (1:hopper.nq)]) - RoboDojo.kinematics_foot(hopper, x1[hopper.nq .+ (1:hopper.nq)])
    ]
end 

function staget_con(x, u, w) 
    vx_stage_max = 1.0
    vx = (x[hopper.nq + 1] - x[1]) / h
    [
    ul - u[collect(1:hopper.nu)]; # control limit (lower)
    u[collect(1:hopper.nu)] - uu; # control limit (upper)
    vx - vx_stage_max;
    -vx - vx_stage_max;
    ]
end 

function terminal_con(x, u, w) 
	x_travel = X_TRAVEL_GOAL
    vx_max = 0.35
    vx_term = (x[hopper.nq + 1] - x[1]) / h
	θ = x[2 * hopper.nq .+ (1:(2 * hopper.nq))]
    [
	x_travel - (x[1] - θ[1])
	x_travel - (x[hopper.nq + 1] - θ[hopper.nq + 1])
    vx_term - vx_max
    -vx_term - vx_max
	x[1:hopper.nq][collect([2, 3, 4])] - θ[1:hopper.nq][collect([2, 3, 4])]
	x[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])] - θ[hopper.nq .+ (1:hopper.nq)][collect([2, 3, 4])]
    ]
end

con1 = iLQR.Constraint(stage1_con, 2 * hopper.nq, hopper.nu + 2 * hopper.nq, idx_ineq=collect(1:6))
cont = iLQR.Constraint(staget_con, 4 * hopper.nq, hopper.nu, idx_ineq=collect(1:6))
conT = iLQR.Constraint(terminal_con, 4 * hopper.nq, 0, idx_ineq=collect(1:4))
cons = [con1, [cont for t = 2:T-1]..., conT];

# ## rollout
ū_stand = let
    u_init = Vector{Vector{Float64}}()
    hover = hopper.gravity * hopper.mass_body * 0.5 * h
    for t = 1:T-1
        α = (t - 1) / (T - 1)
        phase = 2.0 * pi * HOP_CYCLES * α
        leg_push = clamp(hover + 2.2 * max(0.0, sin(phase)), ul[2], uu[2])
        u_t = [0.0; leg_push]
        if t == 1
            u_t = [u_t; x1]
        end
        push!(u_init, u_t)
    end
    u_init
end
x̄, _, _, _, _ = iLQR.rollout(model, x1, ū_stand)
q̄ = state_to_configuration(x̄)
RoboDojo.visualize!(vis, hopper, q̄, Δt=h, fixed_camera=false);

# ## solver
solver = iLQR.solver(model, obj, cons, 
	opts=iLQR.Options(linesearch = :armijo,
		α_min=1.0e-5,
		obj_tol=1.0e-3,
		grad_tol=1.0e-3,
		max_iter=10,
		max_al_iter=15,
		con_tol=0.001,
		ρ_init=1.0, 
		ρ_scale=10.0, 
		verbose=false))
iLQR.initialize_controls!(solver, ū_stand)
iLQR.initialize_states!(solver, x̄);

# ## solve
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver);

@show iLQR.eval_obj(solver.m_data.obj.costs, solver.m_data.x, solver.m_data.u, solver.m_data.w)
@show solver.s_data.iter[1]
@show norm(terminal_con(solver.m_data.x[T], zeros(0), zeros(0))[3:4], Inf)
@show solver.s_data.obj[1] # augmented Lagrangian cost
    
# ## solution
x_sol, u_sol = iLQR.get_trajectory(solver)
q_sol = state_to_configuration(x_sol)
write_hopper_ref_traj_json(
    HOPPER_REF_TRAJ_FILE,
    q_sol,
    u_sol,
    h,
    hopper.nq,
    hopper.nu,
)
println("saved reference json: " * HOPPER_REF_TRAJ_FILE)
RoboDojo.visualize!(vis, hopper, q_sol, Δt=h, fixed_camera=false);

# ## benchmark (NOTE: gate 3 seems to break @benchmark, just run @time instead...)
solver.options.verbose = false
# if Base.find_package("BenchmarkTools") !== nothing
#     @eval using BenchmarkTools
#     @benchmark iLQR.solve!($solver, $x̄, $ū_stand)
# else
#     println("BenchmarkTools not found; running @time once instead.")
#     @time iLQR.solve!(solver, x̄, ū_stand)
# end
