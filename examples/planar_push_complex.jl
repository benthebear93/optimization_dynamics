using OptimizationDynamics
const iLQR = OptimizationDynamics.IterativeLQR

using JLD2
using JSON3
using LinearAlgebra
using OrderedCollections: OrderedDict
using Random
import RoboDojo
using Scratch
using Symbolics

include(joinpath(@__DIR__, "..", "src", "models", "planar_push", "model_v2.jl"))
const planarpush_complex = planarpush_v2

function load_or_codegen_planarpush_complex_residuals(model::PlanarPushV2)
    path = @get_scratch!("planarpush_complex")
    residual_file = joinpath(path, "residual.jld2")
    if isfile(residual_file)
        @load residual_file r_pp_complex_func rz_pp_complex_func rθ_pp_complex_func rz_pp_complex_array rθ_pp_complex_array
        return r_pp_complex_func, rz_pp_complex_func, rθ_pp_complex_func, rz_pp_complex_array, rθ_pp_complex_array
    end

    nq = model.nq
    nu = model.nu
    nz = RoboDojo.num_var(model)
    nθ = 2 * nq + nu + model.nw + 1

    @variables z[1:nz]
    @variables θ[1:nθ]
    @variables κ[1:1]

    r_pp_complex = residual(model, z, θ, κ)
    rz_pp_complex = Symbolics.jacobian(r_pp_complex, z)
    rθ_pp_complex = Symbolics.jacobian(r_pp_complex, θ)

    r_pp_complex_func = build_function(r_pp_complex, z, θ, κ)[2]
    rz_pp_complex_func = build_function(rz_pp_complex, z, θ)[2]
    rθ_pp_complex_func = build_function(rθ_pp_complex, z, θ)[2]
    rz_pp_complex_array = similar(rz_pp_complex, Float64)
    rθ_pp_complex_array = similar(rθ_pp_complex, Float64)
    @save residual_file r_pp_complex_func rz_pp_complex_func rθ_pp_complex_func rz_pp_complex_array rθ_pp_complex_array
    return r_pp_complex_func, rz_pp_complex_func, rθ_pp_complex_func, rz_pp_complex_array, rθ_pp_complex_array
end

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function write_planar_push_ref_traj_json(
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

    nc_ref = isempty(gamma) ? 0 : length(gamma[1])
    nb_ref = isempty(b) ? 0 : length(b[1])
    nz_ref = isempty(z) ? 0 : length(z[1])
    ntheta_ref = isempty(theta) ? 0 : length(theta[1])

    iq0 = collect(1:nq)
    iq1 = collect(nq .+ (1:nq))
    iu1 = collect(2 * nq .+ (1:nu))
    iw1 = collect(2 * nq + nu .+ (1:nw))
    iq2 = collect(1:nq)
    igamma1 = collect(nq .+ (1:nc_ref))
    ib1 = collect(nq + nc_ref .+ (1:nb_ref))

    od = OrderedDict{String,Any}()
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
    od["nc"] = nc_ref
    od["nb"] = nb_ref
    od["nz"] = nz_ref
    od["nθ"] = ntheta_ref

    open(out_file, "w") do io
        JSON3.write(io, od)
    end
    return nothing
end

h = 0.05
T = 26
SHOW_VIS = true
nc = planarpush_complex.nc
nc_impact = planarpush_complex.nc_impact
nw = planarpush_complex.nw

r_pp_complex_func, rz_pp_complex_func, rθ_pp_complex_func, _, _ = load_or_codegen_planarpush_complex_residuals(planarpush_complex)

im_dyn = ImplicitDynamics(
    planarpush_complex,
    h,
    eval(r_pp_complex_func),
    eval(rz_pp_complex_func),
    eval(rθ_pp_complex_func);
    r_tol=1.0e-8,
    κ_eval_tol=5.0e-5,
    κ_grad_tol=1.0e-2,
    nc=1,
    nb=9,
)

nx = 2 * planarpush_complex.nq
nu = planarpush_complex.nu

ilqr_dyn = iLQR.Dynamics(
    (d, x, u, w) -> f(d, im_dyn, x, u, w),
    (dx, x, u, w) -> fx(dx, im_dyn, x, u, w),
    (du, x, u, w) -> fu(du, im_dyn, x, u, w),
    (gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dyn, x, u, w),
    nx,
    nx,
    nu,
    nw,
    nc,
    nc_impact,
)
model = [ilqr_dyn for _ = 1:T-1]

MODE = :rotate
r_contact = planarpush_complex.r_box + planarpush_complex.r_pusher
if MODE == :translate
    q0 = [0.0, 0.0, 0.0, -r_contact - 1.0e-8, 0.0]
    q1 = [0.0, 0.0, 0.0, -r_contact - 1.0e-8, 0.0]
    qT = [0.0, 0.0, 0.0, -r_contact - 1.0e-8, -planarpush_complex.r_box]
    xT = [qT; qT]
else
    q0 = [0.0, 0.0, 0.0, -r_contact - 1.0e-8, -0.01]
    q1 = [0.0, 0.0, 0.0, -r_contact - 1.0e-8, -0.01]
    qT = [0.5, 0.5, 1.5707963267949, 0.5 - planarpush_complex.r_box, 0.5 - planarpush_complex.r_box]
    xT = [qT; qT]
end

function objt(x, u, w)
    q1 = x[1:planarpush_complex.nq]
    q2 = x[planarpush_complex.nq .+ (1:planarpush_complex.nq)]
    v1 = (q2 - q1) ./ h
    J = 0.0
    J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1
    J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT)
    J += 0.5 * (MODE == :translate ? 1.0e-1 : 1.0e-2) * transpose(u) * u
    return J
end

function objT(x, u, w)
    q1 = x[1:planarpush_complex.nq]
    q2 = x[planarpush_complex.nq .+ (1:planarpush_complex.nq)]
    v1 = (q2 - q1) ./ h
    J = 0.0
    J += 0.5 * transpose(v1) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1]) * v1
    J += 0.5 * transpose(x - xT) * Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0, 1.0, 0.1, 0.1]) * (x - xT)
    return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for _ = 1:T-1]..., cT]

ul = [-10.0; -10.0]
uu = [10.0; 10.0]
max_pusher_gap = 0.0001
max_tangent_slip = 0.005
pusher_y_ref = q0[5] - q0[2]

function stage_con(x, u, w)
    nq = planarpush_complex.nq
    q2 = @views x[nq .+ (1:nq)]
    ϕ = ϕ_func(planarpush_complex, q2)
    p_block = q2[1:2]
    Rwb = rot2(q2[3])
    p_local = transpose(Rwb) * (q2[4:5] - p_block)
    slip = p_local[2] - pusher_y_ref
    [
     ul - u;
     u - uu;
     ϕ[1] - max_pusher_gap;
     slip - max_tangent_slip;
     -slip - max_tangent_slip;
    ]
end

function terminal_con(x, u, w)
    [
     (x - xT)[collect([(1:3)..., (6:8)...])];
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu + 3)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for _ = 1:T-1]..., conT]

x1 = [q0; q1]
w = [[0.0] for _ = 1:T]
ū = MODE == :translate ? [t < 5 ? [1.0; 0.0] : [0.0; 0.0] for t = 1:T-1] : [t < 5 ? [1.0; 0.0] : t < 10 ? [0.5; 0.0] : [0.0; 0.0] for t = 1:T-1]
x̄, _, _, _, _ = iLQR.rollout(model, x1, ū, w)
q̄ = state_to_configuration(x̄)

if SHOW_VIS
    vis = Visualizer()
    render(vis)
    visualize!(
        vis,
        planarpush,
        q̄;
        r=planarpush_complex.r_box + planarpush_complex.r_pusher,
        r_pusher=planarpush_complex.r_pusher,
        Δt=h,
    )
end

solver = iLQR.solver(
    model,
    obj,
    cons,
    opts=iLQR.Options(
        linesearch=:armijo,
        α_min=1.0e-5,
        obj_tol=1.0e-3,
        grad_tol=1.0e-3,
        max_iter=10,
        max_al_iter=10,
        con_tol=0.005,
        ρ_init=1.0,
        ρ_scale=10.0,
        verbose=false,
    ),
)
iLQR.initialize_controls!(solver, ū)
iLQR.initialize_states!(solver, x̄)
iLQR.reset!(solver.s_data)
@time iLQR.solve!(solver)

x_sol, u_sol = iLQR.get_trajectory(solver)
x_rollout_sol, gamma_hist, b_hist, ip_z_hist, ip_θ_hist = iLQR.rollout(model, x1, u_sol, w)
q_rollout_sol = state_to_configuration(x_rollout_sol)

if SHOW_VIS
    vis_sol = Visualizer()
    render(vis_sol)
    visualize!(
        vis_sol,
        planarpush,
        q_rollout_sol;
        r=planarpush_complex.r_box + planarpush_complex.r_pusher,
        r_pusher=planarpush_complex.r_pusher,
        Δt=h,
    )
end

out_dir = joinpath(@__DIR__, "..", "data", "reference_trajectory")
mkpath(out_dir)
ref_json_path = joinpath(out_dir, "pusher_ref_traj_complex.json")
write_planar_push_ref_traj_json(
    ref_json_path,
    q_rollout_sol,
    u_sol,
    w,
    gamma_hist,
    b_hist,
    ip_z_hist,
    ip_θ_hist,
    h,
    planarpush_complex.nq,
    planarpush_complex.nu,
    planarpush_complex.nw,
)
println("saved reference json: " * ref_json_path)
