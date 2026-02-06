using LinearAlgebra
using Statistics
using Random
using Plots
using OptimizationDynamics

const iLQR = OptimizationDynamics.IterativeLQR

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "planar_push_data"
mkpath(OUTPUT_DIR)

# Monte Carlo settings
const N_MC = 50
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

function clamp_u(u, ul, uu)
    uc = copy(u)
    for i in eachindex(uc)
        uc[i] = clamp(uc[i], ul[i], uu[i])
    end
    return uc
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
        xnext[nq + 1] += noise_seq[t][1]
        xnext[nq + 2] += noise_seq[t][2]
        xnext[nq + 3] += noise_seq[t][3]
        u_hist[t] = ut
        x_hist[t + 1] = copy(xnext)
        x = xnext
    end
    return x_hist, u_hist
end

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

function trial_metrics(x_hist, u_hist, nq, q_goal)
    q_final = x_hist[end][nq .+ (1:nq)]
    pos_err = q_final[1:2] - q_goal[1:2]
    theta_err = q_final[3] - q_goal[3]
    total_u_mag = sum(norm(u) for u in u_hist)
    success = (norm(pos_err) <= POS_SUCCESS_TOL) && (abs(theta_err) <= THETA_SUCCESS_TOL)
    return norm(pos_err), abs(theta_err), total_u_mag, success
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

rng = MersenneTwister(1234)
N_horizon = length(PointRun.ilqr_dyns)
noise_trials = make_noise_trials(rng, N_MC, N_horizon)

res_point = run_robustness(
    "POINT",
    PointRun.ilqr_dyns,
    PointRun.solver,
    PointRun.x1,
    PointRun.qT,
    PointRun.ul,
    PointRun.uu,
    noise_trials,
)

res_line = run_robustness(
    "LINE",
    LineRun.ilqr_dyns,
    LineRun.solver,
    LineRun.x1,
    LineRun.qT,
    LineRun.ul,
    LineRun.uu,
    noise_trials,
)

# Comparison plots
labels = ["point-open", "point-fb", "line-open", "line-fb"]
pos_mean = [mean(res_point.pos_open), mean(res_point.pos_fb), mean(res_line.pos_open), mean(res_line.pos_fb)]
th_mean = [mean(res_point.th_open), mean(res_point.th_fb), mean(res_line.th_open), mean(res_line.th_fb)]
u_mean = [mean(res_point.u_open), mean(res_point.u_fb), mean(res_line.u_open), mean(res_line.u_fb)]
succ = [mean(res_point.suc_open), mean(res_point.suc_fb), mean(res_line.suc_open), mean(res_line.suc_fb)]

p_pos = bar(labels, pos_mean, label="mean final position error")
savefig(p_pos, joinpath(OUTPUT_DIR, "robustness_pos_mean_bar.png"))

p_th = bar(labels, th_mean, label="mean final abs theta error")
savefig(p_th, joinpath(OUTPUT_DIR, "robustness_theta_mean_bar.png"))

p_u = bar(labels, u_mean, label="mean total sum ||u_t||")
savefig(p_u, joinpath(OUTPUT_DIR, "robustness_total_u_mean_bar.png"))

p_s = bar(labels, succ, label="success rate")
savefig(p_s, joinpath(OUTPUT_DIR, "robustness_success_rate_bar.png"))

println("saved: ", joinpath(OUTPUT_DIR, "robustness_pos_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_theta_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_total_u_mean_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "robustness_success_rate_bar.png"))
