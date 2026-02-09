using LinearAlgebra
using Statistics
using Random
using Plots
using CSV
using DataFrames
using OptimizationDynamics

const iLQR = OptimizationDynamics.IterativeLQR

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "planar_alone_data"
mkpath(OUTPUT_DIR)

# Faster alternative to Monte Carlo:
# one shared disturbance dataset, scaled by scenario.
const NOISE_SEED = something(tryparse(Int, get(ENV, "ALONE_NOISE_SEED", "1234")), 1234)
const NOISE_SEEDS = let s = strip(get(ENV, "ALONE_NOISE_SEEDS", ""))
    if isempty(s)
        [NOISE_SEED]
    else
        [parse(Int, strip(tok)) for tok in split(s, ",") if !isempty(strip(tok))]
    end
end
# Rollout-time pusher friction mismatch surrogate:
# mu_scale = clamp(1 + scale * (MU_PUSHER_SCALE_BIAS + MU_PUSHER_SCALE_STD * randn), min, max)
const MU_PUSHER_SCALE_BIAS = 0.0
const MU_PUSHER_SCALE_STD = 0.05
const MU_PUSHER_SCALE_MIN = 0.80
const MU_PUSHER_SCALE_MAX = 1.20
const NOISE_SCALES = [0.0, 0.5, 1.0, 1.5, 2.0]
const FOCUS_NOISE_SCALE = something(tryparse(Float64, get(ENV, "ALONE_FOCUS_NOISE_SCALE", "1.0")), 1.0)
const FORCE_DIST_SCALE = something(tryparse(Float64, get(ENV, "ALONE_FORCE_DIST_SCALE", "1.0")), 1.0)
const SHOW_MESHCAT_OPEN = lowercase(strip(get(ENV, "ALONE_SHOW_MESHCAT_OPEN", "true"))) in ("1", "true", "yes", "on")
const VIS_NOISE_SCALE = something(tryparse(Float64, get(ENV, "ALONE_VIS_NOISE_SCALE", "1.0")), 1.0)
const VIS_MODEL = lowercase(strip(get(ENV, "ALONE_VIS_MODEL", "line")))
const REPORT_NOISE_SCALE = 1.0
const POS_SUCCESS_TOL = 0.03
const THETA_SUCCESS_TOL = 0.10

module PointRun
include("point_box_moving.jl")
end

module LineRun
include("line_box_moving.jl")
end

function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
end

function rollout_open_loop(dyns, x1, u_nom, w_seq; mu_scale=1.0)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = copy(u_nom[t])
        ut_eff = mu_scale .* ut
        xnext = copy(iLQR.step!(dyns[t], x, ut_eff, w_seq[t]))
        u_hist[t] = ut
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

function rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, ul, uu; mu_scale=1.0)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = u_nom[t] + K[t] * (x - x_nom[t])
        ut = clamp_u(ut, ul, uu)
        ut_eff = mu_scale .* ut
        xnext = copy(iLQR.step!(dyns[t], x, ut_eff, w_seq[t]))
        u_hist[t] = copy(ut)
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

function trial_metrics(x_hist, u_hist, nq, q_goal)
    q_final = x_hist[end][nq .+ (1:nq)]
    pos_err = q_final[1:2] - q_goal[1:2]
    theta_err = q_final[3] - q_goal[3]
    total_u_mag = sum(norm(u) for u in u_hist)
    success = (norm(pos_err) <= POS_SUCCESS_TOL) && (abs(theta_err) <= THETA_SUCCESS_TOL)
    return norm(pos_err), abs(theta_err), total_u_mag, success
end

function contact_metrics(model_label, x_hist, u_hist, w_seq)
    N = min(length(u_hist), length(x_hist) - 1, length(w_seq))
    if N == 0
        return 0.0, 0.0, 0.0
    end

    min_gamma_sum = Inf
    imbalance_acc = 0.0
    friction_util_peak = 0.0

    if model_label == "line"
        γ_buf = zeros(2)
        b_buf = zeros(10)
        μ = LineRun.lineplanarpush_xy.μ_pusher
        for t in 1:N
            LineRun.eval_contact_data!(γ_buf, b_buf, x_hist[t], u_hist[t], w_seq[t])
            g1 = γ_buf[1]
            g2 = γ_buf[2]
            gsum = g1 + g2
            min_gamma_sum = min(min_gamma_sum, gsum)
            imbalance_acc += abs(g1 - g2) / (abs(gsum) + 1.0e-8)
            util1 = abs(b_buf[9]) / (μ * max(abs(g1), 1.0e-8))
            util2 = abs(b_buf[10]) / (μ * max(abs(g2), 1.0e-8))
            friction_util_peak = max(friction_util_peak, max(util1, util2))
        end
    else
        γ_buf = zeros(1)
        b_buf = zeros(9)
        μ = PointRun.planarpush.μ_pusher
        for t in 1:N
            PointRun.eval_contact_data!(γ_buf, b_buf, x_hist[t], u_hist[t], w_seq[t])
            g = γ_buf[1]
            min_gamma_sum = min(min_gamma_sum, g)
            util = abs(b_buf[9]) / (μ * max(abs(g), 1.0e-8))
            friction_util_peak = max(friction_util_peak, util)
        end
    end

    mean_gamma_imbalance = model_label == "line" ? imbalance_acc / N : 0.0
    return min_gamma_sum, mean_gamma_imbalance, friction_util_peak
end

function sample_mu_delta(rng)
    MU_PUSHER_SCALE_BIAS + MU_PUSHER_SCALE_STD * randn(rng)
end

function scaled_mu(base_mu_delta, s)
    clamp(1.0 + s * base_mu_delta, MU_PUSHER_SCALE_MIN, MU_PUSHER_SCALE_MAX)
end

function analyze_model(label, dyns, solver, x1, q_goal, ul, uu, base_mu_delta, scales)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    x_nom, u_nom = iLQR.get_trajectory(solver)
    K = solver.p_data.K

    rows = NamedTuple[]
    for s in scales
        mu_scale = scaled_mu(base_mu_delta, s)

        x_open, u_open = rollout_open_loop(dyns, x1, u_nom, w_seq; mu_scale=mu_scale)
        p_o, t_o, u_o, suc_o = trial_metrics(x_open, u_open, nq, q_goal)
        gmin_o, gimb_o, util_o = contact_metrics(label, x_open, u_open, w_seq)
        push!(rows, (
            model=label, controller="open", noise_scale=s, mu_scale=mu_scale,
            pos_err=p_o, theta_err=t_o, total_u=u_o, success=Float64(suc_o),
            min_gamma_sum=gmin_o, mean_gamma_imbalance=gimb_o, max_friction_util=util_o,
        ))

        x_fb, u_fb = rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, ul, uu; mu_scale=mu_scale)
        p_f, t_f, u_f, suc_f = trial_metrics(x_fb, u_fb, nq, q_goal)
        gmin_f, gimb_f, util_f = contact_metrics(label, x_fb, u_fb, w_seq)
        push!(rows, (
            model=label, controller="feedback", noise_scale=s, mu_scale=mu_scale,
            pos_err=p_f, theta_err=t_f, total_u=u_f, success=Float64(suc_f),
            min_gamma_sum=gmin_f, mean_gamma_imbalance=gimb_f, max_friction_util=util_f,
        ))
    end

    DataFrame(rows)
end

function save_contact_metric_plots(df)
    for ctrl in ["open", "feedback"]
        d = df[df.controller .== ctrl, :]
        g = combine(groupby(d, [:noise_scale, :model]),
            :min_gamma_sum => mean => :min_gamma_sum,
            :max_friction_util => mean => :max_friction_util,
        )
        point = g[g.model .== "point", :]
        line = g[g.model .== "line", :]

        p1 = plot(point.noise_scale, point.min_gamma_sum, marker=:circle, linewidth=2, label="point",
            xlabel="scenario scale", ylabel="min gamma_sum", title="Contact Retention ($(ctrl))")
        plot!(p1, line.noise_scale, line.min_gamma_sum, marker=:circle, linewidth=2, label="line")
        savefig(p1, joinpath(OUTPUT_DIR, "compare_contact_retention_$(ctrl).png"))

        p2 = plot(point.noise_scale, point.max_friction_util, marker=:circle, linewidth=2, label="point",
            xlabel="scenario scale", ylabel="max friction utilization", title="Friction Utilization ($(ctrl))")
        plot!(p2, line.noise_scale, line.max_friction_util, marker=:circle, linewidth=2, label="line")
        savefig(p2, joinpath(OUTPUT_DIR, "compare_friction_util_$(ctrl).png"))
    end
end

function print_feedback_report(df)
    d = df[df.controller .== "feedback", :]
    rep = combine(groupby(d, :model),
        :pos_err => mean => :mean_pos_err,
        :theta_err => mean => :mean_theta_err,
        :total_u => mean => :mean_total_u,
        :success => mean => :success_rate,
        :min_gamma_sum => mean => :mean_min_gamma_sum,
        :max_friction_util => mean => :mean_max_friction_util,
        :mean_gamma_imbalance => mean => :mean_gamma_imbalance,
    )
    CSV.write(joinpath(OUTPUT_DIR, "feedback_report.csv"), rep)
    println("=== Feedback-Centric Report ===")
    show(rep, allrows=true, allcols=true)
    println()
end

function add_seed_column(df, seed)
    df2 = copy(df)
    df2.seed = fill(seed, nrow(df2))
    return df2
end

function save_model_plots(df, model_label)
    df_model = df[df.model .== model_label, :]
    df_open = df_model[df_model.controller .== "open", :]
    df_fb = df_model[df_model.controller .== "feedback", :]

    p1 = plot(df_open.noise_scale, df_open.pos_err, marker=:circle, linewidth=2, label="open",
        xlabel="noise scale", ylabel="final pos error", title="[$model_label] Position Robustness")
    plot!(p1, df_fb.noise_scale, df_fb.pos_err, marker=:circle, linewidth=2, label="feedback")
    savefig(p1, joinpath(OUTPUT_DIR, "$(model_label)_pos_vs_noise.png"))

    p2 = plot(df_open.noise_scale, df_open.theta_err, marker=:circle, linewidth=2, label="open",
        xlabel="noise scale", ylabel="final abs theta error", title="[$model_label] Theta Robustness")
    plot!(p2, df_fb.noise_scale, df_fb.theta_err, marker=:circle, linewidth=2, label="feedback")
    savefig(p2, joinpath(OUTPUT_DIR, "$(model_label)_theta_vs_noise.png"))

    p3 = plot(df_open.noise_scale, df_open.total_u, marker=:circle, linewidth=2, label="open",
        xlabel="noise scale", ylabel="sum ||u_t||", title="[$model_label] Control Effort")
    plot!(p3, df_fb.noise_scale, df_fb.total_u, marker=:circle, linewidth=2, label="feedback")
    savefig(p3, joinpath(OUTPUT_DIR, "$(model_label)_u_vs_noise.png"))
end

function save_cross_model_plots(df)
    point_open = df[(df.model .== "point") .& (df.controller .== "open"), :]
    point_fb = df[(df.model .== "point") .& (df.controller .== "feedback"), :]
    line_open = df[(df.model .== "line") .& (df.controller .== "open"), :]
    line_fb = df[(df.model .== "line") .& (df.controller .== "feedback"), :]

    p_pos = plot(point_open.noise_scale, point_open.pos_err, marker=:circle, linewidth=2, label="point-open",
        xlabel="noise scale", ylabel="final pos error")
    plot!(p_pos, point_fb.noise_scale, point_fb.pos_err, marker=:circle, linewidth=2, label="point-feedback")
    plot!(p_pos, line_open.noise_scale, line_open.pos_err, marker=:circle, linewidth=2, label="line-open")
    plot!(p_pos, line_fb.noise_scale, line_fb.pos_err, marker=:circle, linewidth=2, label="line-feedback")
    title!(p_pos, "Point vs Line Position Error")
    savefig(p_pos, joinpath(OUTPUT_DIR, "compare_point_line_pos_err.png"))

    p_theta = plot(point_open.noise_scale, point_open.theta_err, marker=:circle, linewidth=2, label="point-open",
        xlabel="noise scale", ylabel="final abs theta error")
    plot!(p_theta, point_fb.noise_scale, point_fb.theta_err, marker=:circle, linewidth=2, label="point-feedback")
    plot!(p_theta, line_open.noise_scale, line_open.theta_err, marker=:circle, linewidth=2, label="line-open")
    plot!(p_theta, line_fb.noise_scale, line_fb.theta_err, marker=:circle, linewidth=2, label="line-feedback")
    title!(p_theta, "Point vs Line Theta Error")
    savefig(p_theta, joinpath(OUTPUT_DIR, "compare_point_line_theta_err.png"))

    p_u = plot(point_open.noise_scale, point_open.total_u, marker=:circle, linewidth=2, label="point-open",
        xlabel="noise scale", ylabel="sum ||u_t||")
    plot!(p_u, point_fb.noise_scale, point_fb.total_u, marker=:circle, linewidth=2, label="point-feedback")
    plot!(p_u, line_open.noise_scale, line_open.total_u, marker=:circle, linewidth=2, label="line-open")
    plot!(p_u, line_fb.noise_scale, line_fb.total_u, marker=:circle, linewidth=2, label="line-feedback")
    title!(p_u, "Point vs Line Control Effort")
    savefig(p_u, joinpath(OUTPUT_DIR, "compare_point_line_total_u.png"))
end

function save_cross_model_bar_plots(df)
    for ctrl in ["open", "feedback"]
        d = df[df.controller .== ctrl, :]
        g = combine(
            groupby(d, [:noise_scale, :model]),
            :pos_err => mean => :pos_err,
            :theta_err => mean => :theta_err,
            :total_u => mean => :total_u,
        )
        scales = sort(unique(g.noise_scale))
        x = collect(1:length(scales))
        xt = (x, string.(scales))
        bw = 0.38
        shift = 0.20
        vals(metric, model) = [only(g[(g.noise_scale .== s) .& (g.model .== model), metric]) for s in scales]
        pos_point = vals(:pos_err, "point")
        pos_line = vals(:pos_err, "line")
        theta_point = vals(:theta_err, "point")
        theta_line = vals(:theta_err, "line")
        u_point = vals(:total_u, "point")
        u_line = vals(:total_u, "line")

        p_pos = bar(x .- shift, pos_point, bar_width=bw, label="point", legend=:topright,
            xticks=xt, xlabel="noise scale", ylabel="final pos error",
            title="Point vs Line Position Error ($(ctrl))")
        bar!(p_pos, x .+ shift, pos_line, bar_width=bw, label="line")
        savefig(p_pos, joinpath(OUTPUT_DIR, "bar_point_line_pos_err_$(ctrl).png"))

        p_theta = bar(x .- shift, theta_point, bar_width=bw, label="point", legend=:topright,
            xticks=xt, xlabel="noise scale", ylabel="final abs theta error",
            title="Point vs Line Theta Error ($(ctrl))")
        bar!(p_theta, x .+ shift, theta_line, bar_width=bw, label="line")
        savefig(p_theta, joinpath(OUTPUT_DIR, "bar_point_line_theta_err_$(ctrl).png"))

        p_u = bar(x .- shift, u_point, bar_width=bw, label="point", legend=:topright,
            xticks=xt, xlabel="noise scale", ylabel="sum ||u_t||",
            title="Point vs Line Control Effort ($(ctrl))")
        bar!(p_u, x .+ shift, u_line, bar_width=bw, label="line")
        savefig(p_u, joinpath(OUTPUT_DIR, "bar_point_line_total_u_$(ctrl).png"))
    end
end

function save_focus_scale_seed_plots(df, focus_scale)
    d = df[abs.(df.noise_scale .- focus_scale) .< 1.0e-12, :]
    if nrow(d) == 0
        return
    end
    for ctrl in ["open", "feedback"]
        dc = d[d.controller .== ctrl, :]
        d_point = sort(dc[dc.model .== "point", :], :seed)
        d_line = sort(dc[dc.model .== "line", :], :seed)
        if nrow(d_point) == 0 || nrow(d_line) == 0
            continue
        end
        labels = string.(d_point.seed)

        p = bar(labels, [d_point.pos_err d_line.pos_err],
            label=["point" "line"], bar_position=:dodge, legend=:topright,
            xlabel="noise seed", ylabel="final pos error",
            title="Position Error at noise=$(focus_scale) ($(ctrl))")
        savefig(p, joinpath(OUTPUT_DIR, "seed_sweep_pos_err_noise$(focus_scale)_$(ctrl).png"))
    end
end

function extract_line_force_hist(x_hist, u_hist, w_seq)
    N = min(length(u_hist), length(x_hist) - 1, length(w_seq))
    gamma1 = zeros(N)
    gamma2 = zeros(N)
    gamma_buf = zeros(2)
    for t in 1:N
        LineRun.eval_gamma_contacts!(gamma_buf, x_hist[t], u_hist[t], w_seq[t])
        gamma1[t] = gamma_buf[1]
        gamma2[t] = gamma_buf[2]
    end
    return gamma1, gamma2
end

function save_line_force_distribution_plots(line_dyns, solver, x1, ul, uu, base_mu_delta)
    N = length(line_dyns)
    nq = line_dyns[1].nx ÷ 2
    w_seq = [zeros(line_dyns[t].nw) for t in 1:N]
    x_nom, u_nom = iLQR.get_trajectory(solver)
    K = solver.p_data.K
    mu_scale = scaled_mu(base_mu_delta, FORCE_DIST_SCALE)

    x_open, u_open = rollout_open_loop(line_dyns, x1, u_nom, w_seq; mu_scale=mu_scale)
    x_fb, u_fb = rollout_feedback(line_dyns, x1, u_nom, x_nom, K, w_seq, ul, uu; mu_scale=mu_scale)
    g1_open, g2_open = extract_line_force_hist(x_open, u_open, w_seq)
    g1_fb, g2_fb = extract_line_force_hist(x_fb, u_fb, w_seq)

    t_open = collect(0:(length(g1_open)-1))
    t_fb = collect(0:(length(g1_fb)-1))
    total_open = g1_open .+ g2_open
    total_fb = g1_fb .+ g2_fb
    share_open = g1_open ./ (abs.(total_open) .+ 1.0e-9)
    share_fb = g1_fb ./ (abs.(total_fb) .+ 1.0e-9)

    p_open = plot(t_open, g1_open, linewidth=2, label="gamma1", xlabel="step", ylabel="normal force")
    plot!(p_open, t_open, g2_open, linewidth=2, label="gamma2")
    plot!(p_open, t_open, total_open, linewidth=2, linestyle=:dash, label="gamma_total")
    plot!(p_open, t_open, share_open, linewidth=2, label="gamma1 share", ylabel="force / share")
    title!(p_open, "Line Force Distribution (open, mu-scale scenario=$(FORCE_DIST_SCALE), mu=$(round(mu_scale, digits=3)))")
    savefig(p_open, joinpath(OUTPUT_DIR, "line_force_distribution_open.png"))

    p_fb = plot(t_fb, g1_fb, linewidth=2, label="gamma1", xlabel="step", ylabel="normal force")
    plot!(p_fb, t_fb, g2_fb, linewidth=2, label="gamma2")
    plot!(p_fb, t_fb, total_fb, linewidth=2, linestyle=:dash, label="gamma_total")
    plot!(p_fb, t_fb, share_fb, linewidth=2, label="gamma1 share", ylabel="force / share")
    title!(p_fb, "Line Force Distribution (feedback, mu-scale scenario=$(FORCE_DIST_SCALE), mu=$(round(mu_scale, digits=3)))")
    savefig(p_fb, joinpath(OUTPUT_DIR, "line_force_distribution_feedback.png"))
end

function visualize_open_case(model_label, dyns, solver, x1, base_mu_delta, vis_noise_scale)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    _, u_nom = iLQR.get_trajectory(solver)
    mu_scale = scaled_mu(base_mu_delta, vis_noise_scale)
    x_open, _ = rollout_open_loop(dyns, x1, u_nom, w_seq; mu_scale=mu_scale)
    q_open = state_to_configuration(x_open)

    vis = Visualizer()
    render(vis)
    if model_label == "point"
        visualize!(vis, planarpush, q_open, Δt=PointRun.h)
    else
        visualize!(vis, lineplanarpush_xy, q_open, Δt=LineRun.h)
    end
end

function report_final_positions(label, dyns, solver, x1, q_goal, ul, uu, base_mu_delta, report_noise_scale)
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]
    x_nom, u_nom = iLQR.get_trajectory(solver)
    K = solver.p_data.K
    mu_scale = scaled_mu(base_mu_delta, report_noise_scale)

    x_open, u_open = rollout_open_loop(dyns, x1, u_nom, w_seq; mu_scale=mu_scale)
    x_fb, u_fb = rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, ul, uu; mu_scale=mu_scale)

    qf_open = x_open[end][nq .+ (1:nq)]
    qf_fb = x_fb[end][nq .+ (1:nq)]

    println("[$label] mismatch_scale=$(report_noise_scale), mu_scale=$(mu_scale)")
    println("  goal_xy = ", q_goal[1:2], ", goal_theta = ", q_goal[3])
    println("  open_final_xy = ", qf_open[1:2], ", open_final_theta = ", qf_open[3], ", open_total_u = ", sum(norm(u) for u in u_open))
    println("  fb_final_xy = ", qf_fb[1:2], ", fb_final_theta = ", qf_fb[3], ", fb_total_u = ", sum(norm(u) for u in u_fb))
end

all_runs = DataFrame[]
seed_to_mu_delta = Dict{Int,Float64}()
for seed in NOISE_SEEDS
    rng = MersenneTwister(seed)
    base_mu_delta = sample_mu_delta(rng)
    seed_to_mu_delta[seed] = base_mu_delta

    point_df = analyze_model(
        "point",
        PointRun.ilqr_dyns,
        PointRun.solver,
        PointRun.x1,
        PointRun.qT,
        PointRun.ul,
        PointRun.uu,
        base_mu_delta,
        NOISE_SCALES,
    )

    line_df = analyze_model(
        "line",
        LineRun.ilqr_dyns,
        LineRun.solver,
        LineRun.x1,
        LineRun.qT,
        LineRun.ul,
        LineRun.uu,
        base_mu_delta,
        NOISE_SCALES,
    )

    push!(all_runs, add_seed_column(vcat(point_df, line_df), seed))
end

all_df = vcat(all_runs...)
CSV.write(joinpath(OUTPUT_DIR, "planar_alone_summary.csv"), all_df)

if length(NOISE_SEEDS) == 1
    save_model_plots(all_df, "point")
    save_model_plots(all_df, "line")
    save_cross_model_plots(all_df)
    save_cross_model_bar_plots(all_df)
end
save_focus_scale_seed_plots(all_df, FOCUS_NOISE_SCALE)
save_contact_metric_plots(all_df)
print_feedback_report(all_df)
save_line_force_distribution_plots(
    LineRun.ilqr_dyns,
    LineRun.solver,
    LineRun.x1,
    LineRun.ul,
    LineRun.uu,
    seed_to_mu_delta[NOISE_SEEDS[1]],
)

if SHOW_MESHCAT_OPEN
    base_mu_delta_vis = seed_to_mu_delta[NOISE_SEEDS[1]]
    if VIS_MODEL == "point"
        visualize_open_case("point", PointRun.ilqr_dyns, PointRun.solver, PointRun.x1, base_mu_delta_vis, VIS_NOISE_SCALE)
        println("meshcat: visualized point open-loop at noise scale=", VIS_NOISE_SCALE)
    else
        visualize_open_case("line", LineRun.ilqr_dyns, LineRun.solver, LineRun.x1, base_mu_delta_vis, VIS_NOISE_SCALE)
        println("meshcat: visualized line open-loop at noise scale=", VIS_NOISE_SCALE)
    end
end

base_mu_delta_report = seed_to_mu_delta[NOISE_SEEDS[1]]
report_final_positions("point", PointRun.ilqr_dyns, PointRun.solver, PointRun.x1, PointRun.qT, PointRun.ul, PointRun.uu, base_mu_delta_report, REPORT_NOISE_SCALE)
report_final_positions("line", LineRun.ilqr_dyns, LineRun.solver, LineRun.x1, LineRun.qT, LineRun.ul, LineRun.uu, base_mu_delta_report, REPORT_NOISE_SCALE)

println("saved: ", joinpath(OUTPUT_DIR, "planar_alone_summary.csv"))
println("saved: ", joinpath(OUTPUT_DIR, "point_pos_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "point_theta_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "point_u_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "line_pos_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "line_theta_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "line_u_vs_noise.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_pos_err.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_theta_err.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_total_u.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_pos_err_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_theta_err_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_total_u_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_pos_err_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_theta_err_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "bar_point_line_total_u_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "line_force_distribution_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "line_force_distribution_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "seed_sweep_pos_err_noise$(FOCUS_NOISE_SCALE)_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "seed_sweep_pos_err_noise$(FOCUS_NOISE_SCALE)_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_contact_retention_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_contact_retention_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_friction_util_open.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_friction_util_feedback.png"))
println("saved: ", joinpath(OUTPUT_DIR, "feedback_report.csv"))
