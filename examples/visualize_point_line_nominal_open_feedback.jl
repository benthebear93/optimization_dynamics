using LinearAlgebra
using Random
using OptimizationDynamics
using MeshCat
using Colors

const iLQR = OptimizationDynamics.IterativeLQR

# Disturbance for visual comparison (process noise on block pose after each step)
const DIST_STD_X = 0.002
const DIST_STD_Y = 0.002
const DIST_STD_THETA = 0.005

module PointRun
include("point_push_free_box.jl")
end

module LineRun
include("line_push_free_box.jl")
end

function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
end

function make_noise_seq(rng, N)
    [[DIST_STD_X * randn(rng), DIST_STD_Y * randn(rng), DIST_STD_THETA * randn(rng)] for _ in 1:N]
end

function add_goal_box!(vis, q_goal; name="goal_box", r_box=0.075)
    setobject!(
        vis[name],
        OptimizationDynamics.GeometryBasics.Rect(
            OptimizationDynamics.Vec(-r_box, -r_box, -r_box),
            OptimizationDynamics.Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box),
        ),
        OptimizationDynamics.MeshPhongMaterial(color=RGBA(0.95, 0.1, 0.1, 1.0)),
    )
    settransform!(
        vis[name],
        OptimizationDynamics.compose(
            OptimizationDynamics.Translation(q_goal[1], q_goal[2], 0.0),
            OptimizationDynamics.LinearMap(OptimizationDynamics.RotZ(q_goal[3])),
        ),
    )
end

function rollout_open_loop(dyns, x1, u_nom, noise_seq)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = u_nom[t]
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        xnext[nq + 1] += noise_seq[t][1]
        xnext[nq + 2] += noise_seq[t][2]
        xnext[nq + 3] += noise_seq[t][3]
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist
end

function rollout_feedback(dyns, x1, x_nom, u_nom, K, ul, uu, noise_seq)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = u_nom[t] + K[t] * (x - x_nom[t])
        ut = clamp_u(ut, ul, uu)
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        xnext[nq + 1] += noise_seq[t][1]
        xnext[nq + 2] += noise_seq[t][2]
        xnext[nq + 3] += noise_seq[t][3]
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist
end

function show_three_cases_point_overlay(noise_seq)
    x_nom, u_nom = iLQR.get_trajectory(PointRun.solver)
    K = PointRun.solver.p_data.K

    x_open = rollout_open_loop(PointRun.ilqr_dyns, PointRun.x1, u_nom, noise_seq)
    x_fb = rollout_feedback(PointRun.ilqr_dyns, PointRun.x1, x_nom, u_nom, K, PointRun.ul, PointRun.uu, noise_seq)

    q_nom = state_to_configuration(x_nom)
    q_open = state_to_configuration(x_open)
    q_fb = state_to_configuration(x_fb)

    vis = Visualizer()
    render(vis)
    OptimizationDynamics.default_background!(vis)

    # i=1 nominal, i=2 open-loop disturbed, i=3 feedback disturbed
    OptimizationDynamics._create_planar_push!(vis, planarpush, i=1, tl=0.95,
        box_color=RGBA(0.2, 0.2, 0.2, 0.55),
        pusher_color=RGBA(0.1, 0.4, 0.9, 0.95))
    OptimizationDynamics._create_planar_push!(vis, planarpush, i=2, tl=0.45,
        box_color=RGBA(0.2, 0.2, 0.2, 0.30),
        pusher_color=RGBA(0.9, 0.2, 0.2, 0.45))
    OptimizationDynamics._create_planar_push!(vis, planarpush, i=3, tl=0.45,
        box_color=RGBA(0.2, 0.2, 0.2, 0.30),
        pusher_color=RGBA(0.1, 0.7, 0.2, 0.45))
    add_goal_box!(vis, PointRun.qT, name="goal_box_point")

    Tvis = minimum((length(q_nom), length(q_open), length(q_fb)))
    anim = MeshCat.Animation(convert(Int, floor(1.0 / PointRun.h)))
    for t in 1:(Tvis - 1)
        MeshCat.atframe(anim, t) do
            OptimizationDynamics._set_planar_push!(vis, planarpush, q_nom[t], i=1)
            OptimizationDynamics._set_planar_push!(vis, planarpush, q_open[t], i=2)
            OptimizationDynamics._set_planar_push!(vis, planarpush, q_fb[t], i=3)
        end
    end
    settransform!(vis["/Cameras/default"],
        OptimizationDynamics.compose(
            OptimizationDynamics.Translation(0.0, 0.0, 50.0),
            OptimizationDynamics.LinearMap(
                OptimizationDynamics.RotZ(0.5 * pi) * OptimizationDynamics.RotY(-pi / 2.5),
            ),
        ))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
    MeshCat.setanimation!(vis, anim)

    println("point(one server): blue=nominal, red=open-loop disturbed, green=online-feedback disturbed, solid red box=goal")
end

function show_three_cases_line_overlay(noise_seq)
    x_nom, u_nom = iLQR.get_trajectory(LineRun.solver)
    K = LineRun.solver.p_data.K

    x_open = rollout_open_loop(LineRun.ilqr_dyns, LineRun.x1, u_nom, noise_seq)
    x_fb = rollout_feedback(LineRun.ilqr_dyns, LineRun.x1, x_nom, u_nom, K, LineRun.ul, LineRun.uu, noise_seq)

    q_nom = state_to_configuration(x_nom)
    q_open = state_to_configuration(x_open)
    q_fb = state_to_configuration(x_fb)

    vis = Visualizer()
    render(vis)
    OptimizationDynamics.default_background!(vis)

    OptimizationDynamics._create_planar_push!(vis, lineplanarpush_xy, i=1, tl=0.95,
        box_color=RGBA(0.2, 0.2, 0.2, 0.55),
        pusher_color=RGBA(0.1, 0.4, 0.9, 0.95))
    OptimizationDynamics._create_planar_push!(vis, lineplanarpush_xy, i=2, tl=0.45,
        box_color=RGBA(0.2, 0.2, 0.2, 0.30),
        pusher_color=RGBA(0.9, 0.2, 0.2, 0.45))
    OptimizationDynamics._create_planar_push!(vis, lineplanarpush_xy, i=3, tl=0.45,
        box_color=RGBA(0.2, 0.2, 0.2, 0.30),
        pusher_color=RGBA(0.1, 0.7, 0.2, 0.45))
    add_goal_box!(vis, LineRun.qT, name="goal_box_line")

    Tvis = minimum((length(q_nom), length(q_open), length(q_fb)))
    anim = MeshCat.Animation(convert(Int, floor(1.0 / LineRun.h)))
    for t in 1:(Tvis - 1)
        MeshCat.atframe(anim, t) do
            OptimizationDynamics._set_planar_push!(vis, lineplanarpush_xy, q_nom[t], i=1)
            OptimizationDynamics._set_planar_push!(vis, lineplanarpush_xy, q_open[t], i=2)
            OptimizationDynamics._set_planar_push!(vis, lineplanarpush_xy, q_fb[t], i=3)
        end
    end
    settransform!(vis["/Cameras/default"],
        OptimizationDynamics.compose(
            OptimizationDynamics.Translation(0.0, 0.0, 50.0),
            OptimizationDynamics.LinearMap(
                OptimizationDynamics.RotZ(0.5 * pi) * OptimizationDynamics.RotY(-pi / 2.5),
            ),
        ))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
    MeshCat.setanimation!(vis, anim)

    println("line(one server): blue=nominal, red=open-loop disturbed, green=online-feedback disturbed, solid red box=goal")
end

rng = MersenneTwister(2026)
N = length(PointRun.ilqr_dyns)
noise_seq = make_noise_seq(rng, N)

show_three_cases_point_overlay(noise_seq)
show_three_cases_line_overlay(noise_seq)
