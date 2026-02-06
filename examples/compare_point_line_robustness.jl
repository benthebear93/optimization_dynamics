using LinearAlgebra
using Statistics
using Random
using Plots
using OptimizationDynamics
using CSV
using DataFrames

const iLQR = OptimizationDynamics.IterativeLQR

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "planar_push_data"
mkpath(OUTPUT_DIR)

# Monte Carlo settings
const N_MC = 200
const MC_SEEDS = [1234, 2345, 3456, 4567, 5678]
const DIST_STD_X = 0.0015
const DIST_STD_Y = 0.0015
const DIST_STD_THETA = 0.004
const POS_SUCCESS_TOL = 0.03
const THETA_SUCCESS_TOL = 0.10

module PointRun
include("point_box_moving.jl")
end

module LineRun
include("line_box_moving.jl")
end

# Clamp control input elementwise to lower/upper bounds.
function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
end

# Roll out using nominal controls only (no feedback) under the same disturbance sequence.
function rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = copy(u_nom[t])
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        xnext[nq + 1] += noise_seq[t][1]
        xnext[nq + 2] += noise_seq[t][2]
        xnext[nq + 3] += noise_seq[t][3]
        u_hist[t] = ut
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

# Roll out with time-varying state feedback u = u_nom + K(x - x_nom), with input clamping.
function rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, nq, ul, uu, noise_seq)
    N = length(dyns)
    x_hist = Vector{Vector{Float64}}(undef, N + 1)
    u_hist = Vector{Vector{Float64}}(undef, N)
    x = copy(x1)
    x_hist[1] = copy(x)
    for t in 1:N
        ut = u_nom[t] + K[t] * (x - x_nom[t])
        ut = clamp_u(ut, ul, uu)
        xnext = copy(iLQR.step!(dyns[t], x, ut, w_seq[t]))
        xnext[nq + 1] += noise_seq[t][1]
        xnext[nq + 2] += noise_seq[t][2]
        xnext[nq + 3] += noise_seq[t][3]
        u_hist[t] = copy(ut)
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

# Compute final position/theta errors, total control magnitude, and success flag for one trial.
function trial_metrics(x_hist, u_hist, nq, q_goal)
    q_final = x_hist[end][nq .+ (1:nq)]
    pos_err = q_final[1:2] - q_goal[1:2]
    theta_err = q_final[3] - q_goal[3]
    total_u_mag = sum(norm(u) for u in u_hist)
    success = (norm(pos_err) <= POS_SUCCESS_TOL) && (abs(theta_err) <= THETA_SUCCESS_TOL)
    return norm(pos_err), abs(theta_err), total_u_mag, success
end

# Run Monte Carlo trials for one model and compare open-loop vs feedback statistics.
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
        po, to, uo, so = trial_metrics(xh_o, uh_o, nq, q_goal)
        push!(pos_open, po); push!(th_open, to); push!(u_open, uo); push!(suc_open, so)

        xh_f, uh_f = rollout_feedback(dyns, x1, u_nom, x_nom, K, w_seq, nq, ul, uu, noise_seq)
        pf, tf, uf, sf = trial_metrics(xh_f, uh_f, nq, q_goal)
        push!(pos_fb, pf); push!(th_fb, tf); push!(u_fb, uf); push!(suc_fb, sf)
    end

    println("=== $label Robustness ===")
    println("open-loop: mean pos err = ", mean(pos_open), ", mean |theta err| = ", mean(th_open),
            ", mean total_u = ", mean(u_open), ", success = ", mean(suc_open))
    println("feedback : mean pos err = ", mean(pos_fb), ", mean |theta err| = ", mean(th_fb),
            ", mean total_u = ", mean(u_fb), ", success = ", mean(suc_fb))

    return (
        pos_open=pos_open, th_open=th_open, u_open=u_open, suc_open=suc_open,
        pos_fb=pos_fb, th_fb=th_fb, u_fb=u_fb, suc_fb=suc_fb
    )
end

# Build shared random disturbance trials for fair point-vs-line comparison.
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

N_horizon = length(PointRun.ilqr_dyns)

point_open_pos_mean = Float64[]; point_fb_pos_mean = Float64[]
line_open_pos_mean = Float64[]; line_fb_pos_mean = Float64[]
point_open_th_mean = Float64[]; point_fb_th_mean = Float64[]
line_open_th_mean = Float64[]; line_fb_th_mean = Float64[]
point_open_u_mean = Float64[]; point_fb_u_mean = Float64[]
line_open_u_mean = Float64[]; line_fb_u_mean = Float64[]
point_open_succ = Float64[]; point_fb_succ = Float64[]
line_open_succ = Float64[]; line_fb_succ = Float64[]

all_point_open_pos = Float64[]; all_point_fb_pos = Float64[]
all_line_open_pos = Float64[]; all_line_fb_pos = Float64[]
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
        "POINT(seed=$seed)",
        PointRun.ilqr_dyns,
        PointRun.solver,
        PointRun.x1,
        PointRun.qT,
        PointRun.ul,
        PointRun.uu,
        noise_trials,
    )

    res_line = run_robustness(
        "LINE(seed=$seed)",
        LineRun.ilqr_dyns,
        LineRun.solver,
        LineRun.x1,
        LineRun.qT,
        LineRun.ul,
        LineRun.uu,
        noise_trials,
    )

    push!(point_open_pos_mean, mean(res_point.pos_open)); push!(point_fb_pos_mean, mean(res_point.pos_fb))
    push!(line_open_pos_mean, mean(res_line.pos_open)); push!(line_fb_pos_mean, mean(res_line.pos_fb))
    push!(point_open_th_mean, mean(res_point.th_open)); push!(point_fb_th_mean, mean(res_point.th_fb))
    push!(line_open_th_mean, mean(res_line.th_open)); push!(line_fb_th_mean, mean(res_line.th_fb))
    push!(point_open_u_mean, mean(res_point.u_open)); push!(point_fb_u_mean, mean(res_point.u_fb))
    push!(line_open_u_mean, mean(res_line.u_open)); push!(line_fb_u_mean, mean(res_line.u_fb))
    push!(point_open_succ, mean(res_point.suc_open)); push!(point_fb_succ, mean(res_point.suc_fb))
    push!(line_open_succ, mean(res_line.suc_open)); push!(line_fb_succ, mean(res_line.suc_fb))
    push!(seed_ids, seed)

    append!(all_point_open_pos, res_point.pos_open); append!(all_point_fb_pos, res_point.pos_fb)
    append!(all_line_open_pos, res_line.pos_open); append!(all_line_fb_pos, res_line.pos_fb)
    append!(all_point_open_th, res_point.th_open); append!(all_point_fb_th, res_point.th_fb)
    append!(all_line_open_th, res_line.th_open); append!(all_line_fb_th, res_line.th_fb)
    append!(all_point_open_u, res_point.u_open); append!(all_point_fb_u, res_point.u_fb)
    append!(all_line_open_u, res_line.u_open); append!(all_line_fb_u, res_line.u_fb)
end

function print_seed_aggregate(name, vals)
    println(name, ": mean=", mean(vals), ", std=", std(vals))
end

println("=== Seed-level Aggregate (mean ± std over seeds) ===")
print_seed_aggregate("point-open pos", point_open_pos_mean)
print_seed_aggregate("point-fb pos", point_fb_pos_mean)
print_seed_aggregate("line-open pos", line_open_pos_mean)
print_seed_aggregate("line-fb pos", line_fb_pos_mean)
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

# Save numeric results to CSV for later analysis without re-running.
seed_summary_df = DataFrame(
    seed = seed_ids,
    point_open_pos_mean = point_open_pos_mean,
    point_fb_pos_mean = point_fb_pos_mean,
    line_open_pos_mean = line_open_pos_mean,
    line_fb_pos_mean = line_fb_pos_mean,
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
CSV.write(joinpath(OUTPUT_DIR, "robustness_seed_summary.csv"), seed_summary_df)

agg_summary_df = DataFrame(
    metric = [
        "pos_error", "pos_error", "pos_error", "pos_error",
        "theta_error", "theta_error", "theta_error", "theta_error",
        "total_u", "total_u", "total_u", "total_u",
        "success_rate", "success_rate", "success_rate", "success_rate",
    ],
    method = [
        "point-open", "point-fb", "line-open", "line-fb",
        "point-open", "point-fb", "line-open", "line-fb",
        "point-open", "point-fb", "line-open", "line-fb",
        "point-open", "point-fb", "line-open", "line-fb",
    ],
    mean_over_seeds = [
        mean(point_open_pos_mean), mean(point_fb_pos_mean), mean(line_open_pos_mean), mean(line_fb_pos_mean),
        mean(point_open_th_mean), mean(point_fb_th_mean), mean(line_open_th_mean), mean(line_fb_th_mean),
        mean(point_open_u_mean), mean(point_fb_u_mean), mean(line_open_u_mean), mean(line_fb_u_mean),
        mean(point_open_succ), mean(point_fb_succ), mean(line_open_succ), mean(line_fb_succ),
    ],
    std_over_seeds = [
        std(point_open_pos_mean), std(point_fb_pos_mean), std(line_open_pos_mean), std(line_fb_pos_mean),
        std(point_open_th_mean), std(point_fb_th_mean), std(line_open_th_mean), std(line_fb_th_mean),
        std(point_open_u_mean), std(point_fb_u_mean), std(line_open_u_mean), std(line_fb_u_mean),
        std(point_open_succ), std(point_fb_succ), std(line_open_succ), std(line_fb_succ),
    ],
)
CSV.write(joinpath(OUTPUT_DIR, "robustness_aggregate_summary.csv"), agg_summary_df)

pooled_trials_df = vcat(
    DataFrame(method=fill("point-open", length(all_point_open_pos)), pos_err=all_point_open_pos, theta_err=all_point_open_th, total_u=all_point_open_u),
    DataFrame(method=fill("point-fb", length(all_point_fb_pos)), pos_err=all_point_fb_pos, theta_err=all_point_fb_th, total_u=all_point_fb_u),
    DataFrame(method=fill("line-open", length(all_line_open_pos)), pos_err=all_line_open_pos, theta_err=all_line_open_th, total_u=all_line_open_u),
    DataFrame(method=fill("line-fb", length(all_line_fb_pos)), pos_err=all_line_fb_pos, theta_err=all_line_fb_th, total_u=all_line_fb_u),
)
CSV.write(joinpath(OUTPUT_DIR, "robustness_pooled_trials.csv"), pooled_trials_df)

# Comparison plots
labels = ["point-open", "point-fb", "line-open", "line-fb"]
pos_mean = [mean(all_point_open_pos), mean(all_point_fb_pos), mean(all_line_open_pos), mean(all_line_fb_pos)]
th_mean = [mean(all_point_open_th), mean(all_point_fb_th), mean(all_line_open_th), mean(all_line_fb_th)]
u_mean = [mean(all_point_open_u), mean(all_point_fb_u), mean(all_line_open_u), mean(all_line_fb_u)]
succ = [mean(point_open_succ), mean(point_fb_succ), mean(line_open_succ), mean(line_fb_succ)]

p_pos = bar(labels, pos_mean, label="mean final position error")
savefig(p_pos, joinpath(OUTPUT_DIR, "robustness_pos_mean_bar.png"))

p_th = bar(labels, th_mean, label="mean final abs theta error")
savefig(p_th, joinpath(OUTPUT_DIR, "robustness_theta_mean_bar.png"))

p_u = bar(labels, u_mean, label="mean total sum ||u_t||")
savefig(p_u, joinpath(OUTPUT_DIR, "robustness_total_u_mean_bar.png"))

p_s = bar(labels, succ, label="success rate")
savefig(p_s, joinpath(OUTPUT_DIR, "robustness_success_rate_bar.png"))

# Distribution plots (Monte Carlo)
p_pos_hist = histogram(all_point_open_pos, bins=30, alpha=0.35, label="point-open", xlabel="final pos error", ylabel="count")
histogram!(p_pos_hist, all_point_fb_pos, bins=30, alpha=0.35, label="point-fb")
histogram!(p_pos_hist, all_line_open_pos, bins=30, alpha=0.35, label="line-open")
histogram!(p_pos_hist, all_line_fb_pos, bins=30, alpha=0.35, label="line-fb")
savefig(p_pos_hist, joinpath(OUTPUT_DIR, "robustness_pos_hist.png"))

p_th_hist = histogram(all_point_open_th, bins=30, alpha=0.35, label="point-open", xlabel="final abs theta error", ylabel="count")
histogram!(p_th_hist, all_point_fb_th, bins=30, alpha=0.35, label="point-fb")
histogram!(p_th_hist, all_line_open_th, bins=30, alpha=0.35, label="line-open")
histogram!(p_th_hist, all_line_fb_th, bins=30, alpha=0.35, label="line-fb")
savefig(p_th_hist, joinpath(OUTPUT_DIR, "robustness_theta_hist.png"))

p_u_hist = histogram(all_point_open_u, bins=30, alpha=0.35, label="point-open", xlabel="total sum ||u_t||", ylabel="count")
histogram!(p_u_hist, all_point_fb_u, bins=30, alpha=0.35, label="point-fb")
histogram!(p_u_hist, all_line_open_u, bins=30, alpha=0.35, label="line-open")
histogram!(p_u_hist, all_line_fb_u, bins=30, alpha=0.35, label="line-fb")
savefig(p_u_hist, joinpath(OUTPUT_DIR, "robustness_total_u_hist.png"))

println("saved: ", joinpath(OUTPUT_DIR, "robustness_pos_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_theta_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_total_u_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_success_rate_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_pos_hist.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_theta_hist.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_total_u_hist.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_seed_summary.csv"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_aggregate_summary.csv"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_pooled_trials.csv"))
