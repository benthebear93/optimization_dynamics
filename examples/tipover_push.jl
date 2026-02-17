using OptimizationDynamics
using RoboDojo
using Symbolics
using Scratch
using JLD2
using LinearAlgebra
using Dates
using Plots

const iLQR = OptimizationDynamics.IterativeLQR

include(joinpath(@__DIR__, "..", "src", "models", "tipover_push", "model.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "tipover_push", "simulator.jl"))
include(joinpath(@__DIR__, "..", "src", "models", "tipover_push", "visuals.jl"))

h = 0.05
T = 10
PLOT_RESULTS = true
SHOW_VIS = true
OUTPUT_DIR = joinpath(@__DIR__, "..", "data")
TRAJ_FILE = joinpath(OUTPUT_DIR, "tipover_push_traj.jld2")

# mode:
# - false: optimize + save trajectory
# - true : load saved trajectory and visualize/plot only
VIS_ONLY = false
VIS_ONLY = VIS_ONLY || ("--vis-only" in ARGS)

logmsg(msg) = println("[$(Dates.format(now(), "HH:MM:SS"))] ", msg)

function load_or_codegen_residuals(model::TipOverPush)
    path = @get_scratch!("tipoverpush")
    residual_file = joinpath(path, "residual_symbolics_v2.jld2")

    if isfile(residual_file)
        logmsg("load cached residuals: " * residual_file)
        @load residual_file r_to_func rz_to_func rθ_to_func rz_to_array rθ_to_array
        return r_to_func, rz_to_func, rθ_to_func, rz_to_array, rθ_to_array
    end
    logmsg("no residual cache; start Symbolics codegen")

    nq = model.nq
    nu = model.nu
    nz = num_var(model)
    nθ = 2 * nq + nu + model.nw + 1

    @variables z[1:nz]
    @variables θ[1:nθ]
    @variables κ[1:1]

    r_to = residual(model, z, θ, κ)
    rz_to = Symbolics.jacobian(r_to, z)
    rθ_to = Symbolics.jacobian(r_to, θ)

    r_to_func = build_function(r_to, z, θ, κ)[2]
    rz_to_func = build_function(rz_to, z, θ)[2]
    rθ_to_func = build_function(rθ_to, z, θ)[2]
    rz_to_array = similar(rz_to, Float64)
    rθ_to_array = similar(rθ_to, Float64)

    @save residual_file r_to_func rz_to_func rθ_to_func rz_to_array rθ_to_array
    logmsg("saved residual cache: " * residual_file)
    return r_to_func, rz_to_func, rθ_to_func, rz_to_array, rθ_to_array
end

logmsg("tipover_push start (h=$(h), T=$(T), nq=$(tipoverpush.nq), nu=$(tipoverpush.nu))")
nx = 2 * tipoverpush.nq
nu = tipoverpush.nu

q0 = nominal_configuration(tipoverpush)
q1 = copy(q0)
qT = copy(q0)
qT[1] = 0.04
# For ±90 deg pitch with corner-on-floor contact, box center height is box_half_width.
qT[3] = tipoverpush.box_half_width + 1.0e-8
qT[4] = 0.5 * pi
x1 = [q0; q1]
xT = [qT; qT]

Qv = Diagonal([0.2, 0.1, 0.2, 1.0, 0.05, 0.05, 0.05])
# tip-over task: prioritize pitch tracking; keep x/z as soft regularization only.
Qx = Diagonal([0.2, 0.05, 0.3, 12.0, 0.1, 0.05, 0.2, 0.2, 0.05, 0.3, 12.0, 0.1, 0.05, 0.2])
Ru = 1.0e-2
ϕ_weight = 20.0

function state_parts(x)
    nq = tipoverpush.nq
    q1 = @views x[1:nq]
    q2 = @views x[nq .+ (1:nq)]
    v1 = (q2 - q1) ./ h
    return q1, q2, v1
end

function objt(x, u, w)
    _, q2, v1 = state_parts(x)
    ϕ = ϕ_func(tipoverpush, q2)

    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)
    J += 0.5 * Ru * transpose(u) * u
    J += 0.5 * ϕ_weight * ϕ[5]^2
    return J
end

function objT(x, u, w)
    _, q2, v1 = state_parts(x)
    ϕ = ϕ_func(tipoverpush, q2)
    J = 0.0
    J += 0.5 * transpose(v1) * Qv * v1
    J += 0.5 * transpose(x - xT) * Qx * (x - xT)
    J += 0.5 * ϕ[5]^2
    return J
end

ct = iLQR.Cost(objt, nx, nu)
cT = iLQR.Cost(objT, nx, 0)
obj = [[ct for _ = 1:T-1]..., cT]

ul = [-6.0; -6.0]
uu = [6.0; 6.0]
max_pusher_gap = 5.0e-3

function stage_con(x, u, w)
    _, q2, _ = state_parts(x)
    ϕ = ϕ_func(tipoverpush, q2)
    [
        ul - u
        u - uu
        ϕ[5] - max_pusher_gap
    ]
end

function terminal_con(x, u, w)
    [
        # Hard terminal target on box pitch only.
        x[tipoverpush.nq + 4] - xT[tipoverpush.nq + 4]
    ]
end

cont = iLQR.Constraint(stage_con, nx, nu, idx_ineq=collect(1:(2 * nu + 1)))
conT = iLQR.Constraint(terminal_con, nx, 0)
cons = [[cont for _ = 1:T-1]..., conT]

q_sol = Vector{Vector{Float64}}()
u_sol = Vector{Vector{Float64}}()

if VIS_ONLY
    if !isfile(TRAJ_FILE)
        error("VIS_ONLY=true but trajectory file not found: " * TRAJ_FILE)
    end
    logmsg("load saved trajectory: " * TRAJ_FILE)
    @load TRAJ_FILE q_sol u_sol h_saved qT_saved
    if h_saved != h
        logmsg("warning: loaded h=$(h_saved), current h=$(h)")
    end
    if length(qT_saved) == length(qT)
        qT .= qT_saved
    end
else
    r_to_func, rz_to_func, rθ_to_func, rz_to_array, rθ_to_array = load_or_codegen_residuals(tipoverpush)
    logmsg("residual functions ready")

    logmsg("build ImplicitDynamics")
    im_dyn = ImplicitDynamics(
        tipoverpush,
        h,
        eval(r_to_func),
        eval(rz_to_func),
        eval(rθ_to_func);
        r_tol=1.0e-8,
        κ_eval_tol=1.0e-5,
        κ_grad_tol=1.0e-5,
        nc=tipoverpush.nc,
        nb=2 * tipoverpush.nc,
    )

    nc = tipoverpush.nc
    nc_impact = tipoverpush.nc_impact
    ilqr_dyn = iLQR.Dynamics(
        (d, x, u, w) -> f(d, im_dyn, x, u, w),
        (dx, x, u, w) -> fx(dx, im_dyn, x, u, w),
        (du, x, u, w) -> fu(du, im_dyn, x, u, w),
        (gamma, contact_vel, ip_z, ip_θ, x, u, w) -> f_debug(gamma, contact_vel, ip_z, ip_θ, im_dyn, x, u, w),
        nx,
        nx,
        nu,
        tipoverpush.nw,
        nc,
        nc_impact,
    )
    model = [ilqr_dyn for _ = 1:T-1]
    logmsg("dynamics model ready (nx=$(nx), nu=$(nu), horizon=$(T))")

    w = [zeros(tipoverpush.nw) for _ = 1:T]
    ū = [t <= 5 ? [4.0; 0.0] : t <= 8 ? [1.5; -0.3] : [0.0; 0.0] for t = 1:T-1]
    logmsg("initial rollout start")
    x̄, _, _, _, _ = iLQR.rollout(model, x1, ū, w)
    logmsg("initial rollout done")

    solver = iLQR.solver(
        model,
        obj,
        cons,
        opts=iLQR.Options(
            linesearch=:armijo,
            α_min=1.0e-5,
            obj_tol=1.0e-3,
            grad_tol=1.0e-3,
            max_iter=8,
            max_al_iter=8,
            con_tol=0.01,
            ρ_init=1.0,
            ρ_scale=10.0,
            verbose=false,
        ),
    )

    iLQR.initialize_controls!(solver, ū)
    iLQR.initialize_states!(solver, x̄)
    iLQR.reset!(solver.s_data)
    logmsg("iLQR solve start")
    @time iLQR.solve!(solver)
    logmsg("iLQR solve done")

    x_sol, u_sol = iLQR.get_trajectory(solver)
    q_sol = state_to_configuration(x_sol)
println("final box state q(T): ", q_sol[end][1:4])
println("target box state  : ", qT[1:4])
println("solver iter       : ", solver.s_data.iter[1])
println("final terminal residual (inf-norm): ", norm(terminal_con(x_sol[end], zeros(0), zeros(0)), Inf))
phi_push_all = [ϕ_func(tipoverpush, q)[5] for q in q_sol]
println("min phi_pusher    : ", minimum(phi_push_all))
println("max phi_pusher    : ", maximum(phi_push_all))

    mkpath(OUTPUT_DIR)
    h_saved = h
    qT_saved = copy(qT)
    @save TRAJ_FILE q_sol u_sol h_saved qT_saved
    logmsg("saved trajectory: " * TRAJ_FILE)
end

if PLOT_RESULTS
    mkpath(OUTPUT_DIR)
    ts = collect(0:(length(q_sol) - 1)) .* h
    us = collect(0:(length(u_sol) - 1)) .* h

    box_x = [q[1] for q in q_sol]
    box_z = [q[3] for q in q_sol]
    box_pitch = [q[4] for q in q_sol]
    phi_push = phi_push_all
    ux = [u[1] for u in u_sol]
    uz = [u[2] for u in u_sol]

    p1 = plot(ts, box_pitch, label="pitch", xlabel="time [s]", ylabel="rad", title="Box Pitch")
    hline!(p1, [qT[4]], linestyle=:dash, label="pitch target")

    p2 = plot(ts, box_x, label="x", xlabel="time [s]", ylabel="m", title="Box Position")
    plot!(p2, ts, box_z, label="z")
    hline!(p2, [qT[1]], linestyle=:dash, label="x target")
    hline!(p2, [qT[3]], linestyle=:dot, label="z target")

    p3 = plot(us, ux, label="u_x", xlabel="time [s]", ylabel="input", title="Control")
    plot!(p3, us, uz, label="u_z")

    p4 = plot(ts, phi_push, label="phi_pusher", xlabel="time [s]", ylabel="signed distance", title="Pusher Contact Gap")
    hline!(p4, [max_pusher_gap], linestyle=:dash, label="gap limit")

    plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 700))
    out_png = joinpath(OUTPUT_DIR, "tipover_push_summary.png")
    savefig(plt, out_png)
    println("saved plot: ", out_png)
end

if SHOW_VIS
    logmsg("meshcat visualization start")
    vis = Visualizer()
    render(vis)
    visualize!(vis, tipoverpush, q_sol; Δt=h, cam_zoom=35.0)
    logmsg("meshcat animation loaded")
end
