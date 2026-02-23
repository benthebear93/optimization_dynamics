using LinearAlgebra
using Statistics
using Random
using Plots
using OptimizationDynamics
using CSV
using DataFrames
using MeshCat
using Colors

const iLQR = OptimizationDynamics.IterativeLQR

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "fixed_push_data"
mkpath(OUTPUT_DIR)

# Monte Carlo settings
const N_MC = something(tryparse(Int, get(ENV, "ROBUST_N_MC", "200")), 200)
const MC_SEEDS = begin
    seeds_env = get(ENV, "ROBUST_MC_SEEDS", "")
    if isempty(seeds_env)
        [1234] #, 2345, 3456, 4567]
    else
        [parse(Int, s) for s in split(seeds_env, ",") if !isempty(strip(s))]
    end
end
const DIST_STD_X = something(tryparse(Float64, get(ENV, "ROBUST_DIST_STD_X", "0.0")), 0.0)
const DIST_STD_Y = something(tryparse(Float64, get(ENV, "ROBUST_DIST_STD_Y", "0.0")), 0.0)
const DIST_STD_THETA = something(tryparse(Float64, get(ENV, "ROBUST_DIST_STD_THETA", "0.004")), 0.004)
const SHOW_MESHCAT_FEEDBACK = get(ENV, "ROBUST_SHOW_MESHCAT_FEEDBACK", "true") == "true"
const PLOT_NORMAL_FORCE = get(ENV, "ROBUST_PLOT_NORMAL_FORCE", "true") == "true"
const FB_STOP_AT_GOAL = get(ENV, "ROBUST_FB_STOP_AT_GOAL", "true") == "true"
const FB_THETA_TOL = something(tryparse(Float64, get(ENV, "ROBUST_FB_THETA_TOL", "0.1")), 0.1)

# Fixed planar push tracks rotation only.
const THETA_SUCCESS_TOL = 0.10

module PointRun
include("point_push_fixed_box.jl")
end

module LineRun
include("line_push_fixed_box.jl")
end

function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
end

function add_noise_to_second_config!(xnext, nq, noise)
    if nq == 5
        # Line model: apply symmetric XY disturbance to both pushers and common theta disturbance.
        xnext[nq + 1] += noise[3] # theta
        xnext[nq + 2] += noise[1] # pusher 1 x
        xnext[nq + 3] += noise[2] # pusher 1 y
        xnext[nq + 4] += noise[1] # pusher 2 x
        xnext[nq + 5] += noise[2] # pusher 2 y
    else
        n_inject = min(3, nq)
        for i in 1:n_inject
            xnext[nq + i] += noise[i]
        end
    end
end

function rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = copy(u_nom[t])
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        add_noise_to_second_config!(xnext, nq, noise_seq[t])
        u_hist[t] = ut
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

function rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, nq, ul, uu, noise_seq; goal_theta=nothing)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    goal_reached = false
    for t in 1:N
        if FB_STOP_AT_GOAL && goal_theta !== nothing && (goal_reached || abs(x[nq + 1] - goal_theta) <= FB_THETA_TOL)
            ut = zeros(length(u_nom[t]))
            goal_reached = true
        else
            ut = u_nom[t] + K[t] * (x - x_nom[t])
            ut = clamp_u(ut, ul, uu)
        end
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        add_noise_to_second_config!(xnext, nq, noise_seq[t])
        u_hist[t] = copy(ut)
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

function trial_metrics_fixed(x_hist, u_hist, nq, q_goal)
    q_final = x_hist[end][nq .+ (1:nq)]
    theta_err = q_final[1] - q_goal[1]
    total_u_mag = sum(norm(u) for u in u_hist)
    success = abs(theta_err) <= THETA_SUCCESS_TOL
    return 0.0, abs(theta_err), total_u_mag, success
end

function run_robustness(label, dyns, solver, x1, q_goal, ul, uu, noise_trials)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    x_nom, u_nom = iLQR.get_trajectory(solver)
    K = solver.p_data.K

    pos_open = Float64[]
    th_open = Float64[]
    u_open = Float64[]
    suc_open = Bool[]

    pos_fb = Float64[]
    th_fb = Float64[]
    u_fb = Float64[]
    suc_fb = Bool[]

    for k in 1:length(noise_trials)
        noise_seq = noise_trials[k]
        xh_o, uh_o = rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
        po, to, uo, so = trial_metrics_fixed(xh_o, uh_o, nq, q_goal)
        push!(pos_open, po); push!(th_open, to); push!(u_open, uo); push!(suc_open, so)

        xh_f, uh_f = rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, nq, ul, uu, noise_seq; goal_theta=q_goal[1])
        pf, tf, uf, sf = trial_metrics_fixed(xh_f, uh_f, nq, q_goal)
        push!(pos_fb, pf); push!(th_fb, tf); push!(u_fb, uf); push!(suc_fb, sf)
    end

    println("=== $label Robustness ===")
    println("open-loop: mean |theta err| = ", mean(th_open),
            ", mean total_u = ", mean(u_open), ", success = ", mean(suc_open))
    println("feedback : mean |theta err| = ", mean(th_fb),
            ", mean total_u = ", mean(u_fb), ", success = ", mean(suc_fb))

    return (
        pos_open=pos_open, th_open=th_open, u_open=u_open, suc_open=suc_open,
        pos_fb=pos_fb, th_fb=th_fb, u_fb=u_fb, suc_fb=suc_fb
    )
end

function make_noise_trials(rng, N_mc, N_horizon)
    trials = Vector{Vector{Vector{Float64}}}(undef, N_mc)
    for k in 1:N_mc
        seq = Vector{Vector{Float64}}(undef, N_horizon)
        for t in 1:N_horizon
            seq[t] = [
                DIST_STD_X * randn(rng),
                DIST_STD_Y * randn(rng),
                DIST_STD_THETA * randn(rng),
            ]
        end
        trials[k] = seq
    end
    return trials
end

function visualize_open_feedback_meshcat(
    label,
    model,
    dyns,
    solver,
    x1,
    ul,
    uu,
    noise_seq;
    goal_theta=nothing,
    h=0.05,
)
    x_nom, u_nom = iLQR.get_trajectory(solver)
    K = solver.p_data.K
    w_seq = [zeros(dyns[t].nw) for t in 1:length(dyns)]
    nq = dyns[1].nx ÷ 2
    x_open, _ = rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
    x_fb, _ = rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, nq, ul, uu, noise_seq; goal_theta=goal_theta)
    q_open = state_to_configuration(x_open)
    q_fb = state_to_configuration(x_fb)

    vis = Visualizer()
    render(vis)
    OptimizationDynamics.default_background!(vis)
    OptimizationDynamics._create_planar_push!(
        vis,
        model,
        i=1, # open-loop disturbed
        tl=0.6,
        box_color=RGBA(0.2, 0.2, 0.2, 0.6),
        pusher_color=RGBA(0.9, 0.2, 0.2, 0.9),
    )
    OptimizationDynamics._create_planar_push!(
        vis,
        model,
        i=2, # feedback disturbed
        tl=0.95,
        box_color=RGBA(0.2, 0.2, 0.2, 0.6),
        pusher_color=RGBA(0.1, 0.7, 0.2, 0.95),
    )

    anim = MeshCat.Animation(convert(Int, floor(1.0 / h)))
    Tvis = min(length(q_open), length(q_fb))
    for t in 1:(Tvis - 1)
        MeshCat.atframe(anim, t) do
            OptimizationDynamics._set_planar_push!(vis, model, q_open[t], i=1)
            OptimizationDynamics._set_planar_push!(vis, model, q_fb[t], i=2)
        end
    end

    settransform!(
        vis["/Cameras/default"],
        OptimizationDynamics.compose(
            OptimizationDynamics.Translation(0.0, 0.0, 50.0),
            OptimizationDynamics.LinearMap(
                OptimizationDynamics.RotZ(0.5 * pi) * OptimizationDynamics.RotY(-pi / 2.5),
            ),
        ),
    )
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)
    MeshCat.setanimation!(vis, anim)
    println("MeshCat visualization opened for $label (red=open-loop, green=feedback)")
    return vis
end

function plot_normal_force_over_time(label, dyns, x1, u_open, u_fb, h)
    N = length(dyns)
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    _, gamma_open = iLQR.rollout(dyns, x1, u_open, w_seq)
    _, gamma_fb = iLQR.rollout(dyns, x1, u_fb, w_seq)

    nc = length(gamma_open[1])
    time_controls = collect(0:h:(N - 1) * h)
    p = plot(
        xlabel="Time (s)",
        ylabel="Normal force",
        title="[$label] Normal Force vs Time",
    )

    for c in 1:nc
        g_open_c = [gamma_open[t][c] for t in 1:N]
        g_fb_c = [gamma_fb[t][c] for t in 1:N]
        plot!(p, time_controls, g_open_c, label="open γ$c", linewidth=2, linestyle=:dash)
        plot!(p, time_controls, g_fb_c, label="fb γ$c", linewidth=2)
    end

    g_open_sum = [sum(gamma_open[t]) for t in 1:N]
    g_fb_sum = [sum(gamma_fb[t]) for t in 1:N]
    plot!(p, time_controls, g_open_sum, label="open γ_sum", linewidth=3, linestyle=:dot)
    plot!(p, time_controls, g_fb_sum, label="fb γ_sum", linewidth=3)

    out = joinpath(OUTPUT_DIR, "normal_force_$(lowercase(label))_open_vs_fb.png")
    savefig(p, out)
    println("saved: ", out)
end

N_horizon = length(PointRun.ilqr_dyns)

point_open_th_mean = Float64[]; point_fb_th_mean = Float64[]
line_open_th_mean = Float64[]; line_fb_th_mean = Float64[]
point_open_u_mean = Float64[]; point_fb_u_mean = Float64[]
line_open_u_mean = Float64[]; line_fb_u_mean = Float64[]
point_open_succ = Float64[]; point_fb_succ = Float64[]
line_open_succ = Float64[]; line_fb_succ = Float64[]

all_point_open_th = Float64[]; all_point_fb_th = Float64[]
all_line_open_th = Float64[]; all_line_fb_th = Float64[]
all_point_open_u = Float64[]; all_point_fb_u = Float64[]
all_line_open_u = Float64[]; all_line_fb_u = Float64[]

seed_ids = Int[]

for seed in MC_SEEDS
    println("=== Running seed $seed (N_MC=$N_MC) ===")
    rng = MersenneTwister(seed)
    noise_trials = make_noise_trials(rng, N_MC, N_horizon)

    res_point = run_robustness(
        "POINT_FIXED(seed=$seed)",
        PointRun.ilqr_dyns,
        PointRun.solver,
        PointRun.x1,
        PointRun.qT,
        PointRun.ul,
        PointRun.uu,
        noise_trials,
    )

    res_line = run_robustness(
        "LINE_FIXED(seed=$seed)",
        LineRun.ilqr_dyns,
        LineRun.solver,
        LineRun.x1,
        LineRun.qT,
        LineRun.ul,
        LineRun.uu,
        noise_trials,
    )

    push!(point_open_th_mean, mean(res_point.th_open)); push!(point_fb_th_mean, mean(res_point.th_fb))
    push!(line_open_th_mean, mean(res_line.th_open)); push!(line_fb_th_mean, mean(res_line.th_fb))
    push!(point_open_u_mean, mean(res_point.u_open)); push!(point_fb_u_mean, mean(res_point.u_fb))
    push!(line_open_u_mean, mean(res_line.u_open)); push!(line_fb_u_mean, mean(res_line.u_fb))
    push!(point_open_succ, mean(res_point.suc_open)); push!(point_fb_succ, mean(res_point.suc_fb))
    push!(line_open_succ, mean(res_line.suc_open)); push!(line_fb_succ, mean(res_line.suc_fb))
    push!(seed_ids, seed)

    append!(all_point_open_th, res_point.th_open); append!(all_point_fb_th, res_point.th_fb)
    append!(all_line_open_th, res_line.th_open); append!(all_line_fb_th, res_line.th_fb)
    append!(all_point_open_u, res_point.u_open); append!(all_point_fb_u, res_point.u_fb)
    append!(all_line_open_u, res_line.u_open); append!(all_line_fb_u, res_line.u_fb)
end

function print_seed_aggregate(name, vals)
    println(name, ": mean=", mean(vals), ", std=", std(vals))
end

println("=== Seed-level Aggregate (mean ± std over seeds) ===")
print_seed_aggregate("point-open th", point_open_th_mean)
print_seed_aggregate("point-fb th", point_fb_th_mean)
print_seed_aggregate("line-open th", line_open_th_mean)
print_seed_aggregate("line-fb th", line_fb_th_mean)
print_seed_aggregate("point-open total_u", point_open_u_mean)
print_seed_aggregate("point-fb total_u", point_fb_u_mean)
print_seed_aggregate("line-open total_u", line_open_u_mean)
print_seed_aggregate("line-fb total_u", line_fb_u_mean)
print_seed_aggregate("point-open success", point_open_succ)
print_seed_aggregate("point-fb success", point_fb_succ)
print_seed_aggregate("line-open success", line_open_succ)
print_seed_aggregate("line-fb success", line_fb_succ)

labels = ["point-open", "point-fb", "line-open", "line-fb"]
th_mean = [mean(all_point_open_th), mean(all_point_fb_th), mean(all_line_open_th), mean(all_line_fb_th)]
u_mean = [mean(all_point_open_u), mean(all_point_fb_u), mean(all_line_open_u), mean(all_line_fb_u)]
succ = [mean(point_open_succ), mean(point_fb_succ), mean(line_open_succ), mean(line_fb_succ)]

p_th = bar(labels, th_mean, label="mean final abs theta error")
savefig(p_th, joinpath(OUTPUT_DIR, "fixed_robustness_theta_mean_bar.png"))

p_u = bar(labels, u_mean, label="mean total sum ||u_t||")
savefig(p_u, joinpath(OUTPUT_DIR, "fixed_robustness_total_u_mean_bar.png"))

p_s = bar(labels, succ, label="success rate")
savefig(p_s, joinpath(OUTPUT_DIR, "fixed_robustness_success_rate_bar.png"))

p_th_hist = histogram(all_point_open_th, bins=30, alpha=0.35, label="point-open", xlabel="final abs theta error", ylabel="count")
histogram!(p_th_hist, all_point_fb_th, bins=30, alpha=0.35, label="point-fb")
histogram!(p_th_hist, all_line_open_th, bins=30, alpha=0.35, label="line-open")
histogram!(p_th_hist, all_line_fb_th, bins=30, alpha=0.35, label="line-fb")
savefig(p_th_hist, joinpath(OUTPUT_DIR, "fixed_robustness_theta_hist.png"))

p_u_hist = histogram(all_point_open_u, bins=30, alpha=0.35, label="point-open", xlabel="total sum ||u_t||", ylabel="count")
histogram!(p_u_hist, all_point_fb_u, bins=30, alpha=0.35, label="point-fb")
histogram!(p_u_hist, all_line_open_u, bins=30, alpha=0.35, label="line-open")
histogram!(p_u_hist, all_line_fb_u, bins=30, alpha=0.35, label="line-fb")
savefig(p_u_hist, joinpath(OUTPUT_DIR, "fixed_robustness_total_u_hist.png"))

seed_summary_df = DataFrame(
    seed = seed_ids,
    point_open_th_mean = point_open_th_mean,
    point_fb_th_mean = point_fb_th_mean,
    line_open_th_mean = line_open_th_mean,
    line_fb_th_mean = line_fb_th_mean,
    point_open_u_mean = point_open_u_mean,
    point_fb_u_mean = point_fb_u_mean,
    line_open_u_mean = line_open_u_mean,
    line_fb_u_mean = line_fb_u_mean,
    point_open_success = point_open_succ,
    point_fb_success = point_fb_succ,
    line_open_success = line_open_succ,
    line_fb_success = line_fb_succ,
)
CSV.write(joinpath(OUTPUT_DIR, "fixed_robustness_seed_summary.csv"), seed_summary_df)

agg_summary_df = DataFrame(
    metric = [
        "theta_error", "theta_error", "theta_error", "theta_error",
        "total_u", "total_u", "total_u", "total_u",
        "success_rate", "success_rate", "success_rate", "success_rate",
    ],
    method = [
        "point-open", "point-fb", "line-open", "line-fb",
        "point-open", "point-fb", "line-open", "line-fb",
        "point-open", "point-fb", "line-open", "line-fb",
    ],
    mean_over_seeds = [
        mean(point_open_th_mean), mean(point_fb_th_mean), mean(line_open_th_mean), mean(line_fb_th_mean),
        mean(point_open_u_mean), mean(point_fb_u_mean), mean(line_open_u_mean), mean(line_fb_u_mean),
        mean(point_open_succ), mean(point_fb_succ), mean(line_open_succ), mean(line_fb_succ),
    ],
    std_over_seeds = [
        std(point_open_th_mean), std(point_fb_th_mean), std(line_open_th_mean), std(line_fb_th_mean),
        std(point_open_u_mean), std(point_fb_u_mean), std(line_open_u_mean), std(line_fb_u_mean),
        std(point_open_succ), std(point_fb_succ), std(line_open_succ), std(line_fb_succ),
    ],
)
CSV.write(joinpath(OUTPUT_DIR, "fixed_robustness_aggregate_summary.csv"), agg_summary_df)

pooled_trials_df = vcat(
    DataFrame(method=fill("point-open", length(all_point_open_th)), theta_err=all_point_open_th, total_u=all_point_open_u),
    DataFrame(method=fill("point-fb", length(all_point_fb_th)), theta_err=all_point_fb_th, total_u=all_point_fb_u),
    DataFrame(method=fill("line-open", length(all_line_open_th)), theta_err=all_line_open_th, total_u=all_line_open_u),
    DataFrame(method=fill("line-fb", length(all_line_fb_th)), theta_err=all_line_fb_th, total_u=all_line_fb_u),
)
CSV.write(joinpath(OUTPUT_DIR, "fixed_robustness_pooled_trials.csv"), pooled_trials_df)

println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_theta_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_total_u_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_success_rate_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_theta_hist.png"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_total_u_hist.png"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_seed_summary.csv"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_aggregate_summary.csv"))
println("saved: ", joinpath(OUTPUT_DIR, "fixed_robustness_pooled_trials.csv"))

if SHOW_MESHCAT_FEEDBACK || PLOT_NORMAL_FORCE
    vis_rng = MersenneTwister(first(MC_SEEDS))
    vis_noise_seq = make_noise_trials(vis_rng, 1, N_horizon)[1]
    point_x_nom, point_u_nom = iLQR.get_trajectory(PointRun.solver)
    line_x_nom, line_u_nom = iLQR.get_trajectory(LineRun.solver)
    point_w_seq = [zeros(PointRun.ilqr_dyns[t].nw) for t in 1:length(PointRun.ilqr_dyns)]
    line_w_seq = [zeros(LineRun.ilqr_dyns[t].nw) for t in 1:length(LineRun.ilqr_dyns)]
    point_nq = PointRun.ilqr_dyns[1].nx ÷ 2
    line_nq = LineRun.ilqr_dyns[1].nx ÷ 2
    _, point_u_fb = rollout_feedback(
        PointRun.ilqr_dyns,
        PointRun.x1,
        point_u_nom,
        point_x_nom,
        PointRun.solver.p_data.K,
        point_w_seq,
        point_nq,
        PointRun.ul,
        PointRun.uu,
        vis_noise_seq,
        goal_theta=PointRun.qT[1],
    )
    _, line_u_fb = rollout_feedback(
        LineRun.ilqr_dyns,
        LineRun.x1,
        line_u_nom,
        line_x_nom,
        LineRun.solver.p_data.K,
        line_w_seq,
        line_nq,
        LineRun.ul,
        LineRun.uu,
        vis_noise_seq,
        goal_theta=LineRun.qT[1],
    )

    if SHOW_MESHCAT_FEEDBACK
        vis_point = visualize_open_feedback_meshcat(
            "point",
            PointRun.fixedplanarpush,
            PointRun.ilqr_dyns,
            PointRun.solver,
            PointRun.x1,
        PointRun.ul,
        PointRun.uu,
        vis_noise_seq;
        goal_theta=PointRun.qT[1],
        h=PointRun.h,
    )
        vis_line = visualize_open_feedback_meshcat(
            "line",
            LineRun.lineplanarpush,
            LineRun.ilqr_dyns,
            LineRun.solver,
            LineRun.x1,
        LineRun.ul,
        LineRun.uu,
        vis_noise_seq;
        goal_theta=LineRun.qT[1],
        h=LineRun.h,
    )
    end

    if PLOT_NORMAL_FORCE
        plot_normal_force_over_time("point", PointRun.ilqr_dyns, PointRun.x1, point_u_nom, point_u_fb, PointRun.h)
        plot_normal_force_over_time("line", LineRun.ilqr_dyns, LineRun.x1, line_u_nom, line_u_fb, LineRun.h)
    end
end
