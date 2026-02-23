using LinearAlgebra
using Statistics
using Random
using CSV
using DataFrames
using Plots
using OptimizationDynamics

const iLQR = OptimizationDynamics.IterativeLQR
ENV["GKSwstype"] = "100"

const OUTDIR = "planar_alone_data/line_branch_diagnosis"
mkpath(OUTDIR)

# Configuration
const NOISE_SEED = something(tryparse(Int, get(ENV, "LINE_BRANCH_NOISE_SEED", "1234")), 1234)
const DIST_STD_X = something(tryparse(Float64, get(ENV, "LINE_BRANCH_DIST_STD_X", "0.0015")), 0.0015)
const DIST_STD_Y = something(tryparse(Float64, get(ENV, "LINE_BRANCH_DIST_STD_Y", "0.0015")), 0.0015)
const DIST_STD_THETA = something(tryparse(Float64, get(ENV, "LINE_BRANCH_DIST_STD_THETA", "0.004")), 0.004)
const COMPARE_SCALES = [0.5, 1.0, 1.5]
const SWEEP_SCALES = collect(0.8:0.02:1.2)
const PERTURB_SCALE = 1.0
const PERTURB_TRIALS = something(tryparse(Int, get(ENV, "LINE_BRANCH_PERTURB_TRIALS", "10")), 10)
const PERTURB_STD = something(tryparse(Float64, get(ENV, "LINE_BRANCH_PERTURB_STD", "1.0e-5")), 1.0e-5)

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

function make_base_noise(rng, N)
    [[
        DIST_STD_X * randn(rng),
        DIST_STD_Y * randn(rng),
        DIST_STD_THETA * randn(rng),
    ] for _ in 1:N]
end

function scaled_noise(base_noise, s)
    [[s * n[1], s * n[2], s * n[3]] for n in base_noise]
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

function rot2(θ)
    [cos(θ) -sin(θ); sin(θ) cos(θ)]
end

function collect_step_metrics(x_hist, u_hist, w_seq, h)
    nq = LineRun.lineplanarpush_xy.nq
    q_hist = state_to_configuration(x_hist)
    N = min(length(u_hist), length(q_hist) - 1)

    gamma1 = zeros(N)
    gamma2 = zeros(N)
    gamma_sum = zeros(N)
    phi1 = zeros(N)
    phi2 = zeros(N)
    phi_max = zeros(N)
    slip1 = zeros(N)
    slip2 = zeros(N)
    slip_max_abs = zeros(N)
    theta = zeros(N)

    gamma_buf = zeros(2)
    for t in 1:N
        q_prev = q_hist[t]
        q_curr = q_hist[t + 1]

        ϕ = ϕ_func(LineRun.lineplanarpush_xy, q_curr)
        phi1[t] = ϕ[1]
        phi2[t] = ϕ[2]
        phi_max[t] = max(ϕ[1], ϕ[2])

        p_block_prev = q_prev[1:2]
        p_block_curr = q_curr[1:2]

        p11_local_prev = transpose(rot2(q_prev[3])) * (q_prev[4:5] - p_block_prev)
        p12_local_prev = transpose(rot2(q_prev[3])) * (q_prev[6:7] - p_block_prev)
        p21_local = transpose(rot2(q_curr[3])) * (q_curr[4:5] - p_block_curr)
        p22_local = transpose(rot2(q_curr[3])) * (q_curr[6:7] - p_block_curr)

        slip1[t] = (p21_local[2] - p11_local_prev[2]) / h
        slip2[t] = (p22_local[2] - p12_local_prev[2]) / h
        slip_max_abs[t] = max(abs(slip1[t]), abs(slip2[t]))

        LineRun.eval_gamma_contacts!(gamma_buf, x_hist[t], u_hist[t], w_seq[t])
        gamma1[t] = gamma_buf[1]
        gamma2[t] = gamma_buf[2]
        gamma_sum[t] = gamma1[t] + gamma2[t]
        theta[t] = q_curr[3]
    end

    DataFrame(
        step=collect(1:N),
        time=(collect(1:N) .* h),
        gamma1=gamma1,
        gamma2=gamma2,
        gamma_sum=gamma_sum,
        phi1=phi1,
        phi2=phi2,
        phi_max=phi_max,
        slip1=slip1,
        slip2=slip2,
        slip_max_abs=slip_max_abs,
        theta=theta,
    )
end

function trial_summary(x_hist, u_hist, q_goal)
    nq = LineRun.lineplanarpush_xy.nq
    q_final = x_hist[end][nq .+ (1:nq)]
    pos_err = norm(q_final[1:2] - q_goal[1:2])
    theta_err = abs(q_final[3] - q_goal[3])
    total_u = sum(norm(u) for u in u_hist)
    return pos_err, theta_err, total_u
end

function save_scale_comparison_plots(scale_to_df)
    scales = sort(collect(keys(scale_to_df)))

    p1 = plot(xlabel="time (s)", ylabel="gamma_sum", title="gamma_sum by noise scale")
    p2 = plot(xlabel="time (s)", ylabel="phi_max", title="phi_max by noise scale")
    p3 = plot(xlabel="time (s)", ylabel="slip_max_abs", title="slip_max_abs by noise scale")
    p4 = plot(xlabel="time (s)", ylabel="theta", title="theta by noise scale")

    for s in scales
        d = scale_to_df[s]
        lbl = "scale=$(s)"
        plot!(p1, d.time, d.gamma_sum, label=lbl, linewidth=2)
        plot!(p2, d.time, d.phi_max, label=lbl, linewidth=2)
        plot!(p3, d.time, d.slip_max_abs, label=lbl, linewidth=2)
        plot!(p4, d.time, d.theta, label=lbl, linewidth=2)
    end

    savefig(p1, joinpath(OUTDIR, "compare_gamma_sum_scales.png"))
    savefig(p2, joinpath(OUTDIR, "compare_phi_max_scales.png"))
    savefig(p3, joinpath(OUTDIR, "compare_slip_scales.png"))
    savefig(p4, joinpath(OUTDIR, "compare_theta_scales.png"))
end

function run()
    dyns = LineRun.ilqr_dyns
    solver = LineRun.solver
    x1 = LineRun.x1
    q_goal = LineRun.qT
    N = length(dyns)
    nq = dyns[1].nx ÷ 2
    w_seq = [zeros(dyns[t].nw) for t in 1:N]

    x_nom, u_nom = iLQR.get_trajectory(solver)
    rng = MersenneTwister(NOISE_SEED)
    base_noise = make_base_noise(rng, N)

    # (2) step-wise signal comparison for selected scales
    scale_to_df = Dict{Float64,DataFrame}()
    summary_rows = NamedTuple[]
    for s in COMPARE_SCALES
        noise_seq = scaled_noise(base_noise, s)
        x_open, u_open = rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
        d = collect_step_metrics(x_open, u_open, w_seq, LineRun.h)
        d.noise_scale = fill(s, nrow(d))
        CSV.write(joinpath(OUTDIR, "timeseries_open_noise_$(s).csv"), d)
        scale_to_df[s] = d

        pos_err, theta_err, total_u = trial_summary(x_open, u_open, q_goal)
        push!(summary_rows, (
            mode="compare_scale",
            noise_scale=s,
            pos_err=pos_err,
            theta_err=theta_err,
            total_u=total_u,
            min_gamma_sum=minimum(d.gamma_sum),
            max_phi=maximum(vcat(d.phi1, d.phi2)),
            max_slip=maximum(d.slip_max_abs),
        ))
    end
    save_scale_comparison_plots(scale_to_df)

    # (3) local sweep around the branch
    for s in SWEEP_SCALES
        noise_seq = scaled_noise(base_noise, s)
        x_open, u_open = rollout_open_loop(dyns, x1, u_nom, w_seq, nq, noise_seq)
        d = collect_step_metrics(x_open, u_open, w_seq, LineRun.h)
        pos_err, theta_err, total_u = trial_summary(x_open, u_open, q_goal)
        push!(summary_rows, (
            mode="sweep",
            noise_scale=s,
            pos_err=pos_err,
            theta_err=theta_err,
            total_u=total_u,
            min_gamma_sum=minimum(d.gamma_sum),
            max_phi=maximum(vcat(d.phi1, d.phi2)),
            max_slip=maximum(d.slip_max_abs),
        ))
    end

    # (4) tiny initial-state perturbation sensitivity at scale=1.0
    perturb_rng = MersenneTwister(NOISE_SEED + 99)
    noise_seq_pert = scaled_noise(base_noise, PERTURB_SCALE)
    for k in 1:PERTURB_TRIALS
        x1p = copy(x1)
        dx = PERTURB_STD * randn(perturb_rng)
        dy = PERTURB_STD * randn(perturb_rng)
        dθ = PERTURB_STD * randn(perturb_rng)
        # Perturb both q1/q2 so initial velocity does not jump.
        x1p[1] += dx
        x1p[2] += dy
        x1p[3] += dθ
        x1p[nq + 1] += dx
        x1p[nq + 2] += dy
        x1p[nq + 3] += dθ

        x_open, u_open = rollout_open_loop(dyns, x1p, u_nom, w_seq, nq, noise_seq_pert)
        d = collect_step_metrics(x_open, u_open, w_seq, LineRun.h)
        pos_err, theta_err, total_u = trial_summary(x_open, u_open, q_goal)
        push!(summary_rows, (
            mode="perturb",
            noise_scale=PERTURB_SCALE,
            pos_err=pos_err,
            theta_err=theta_err,
            total_u=total_u,
            min_gamma_sum=minimum(d.gamma_sum),
            max_phi=maximum(vcat(d.phi1, d.phi2)),
            max_slip=maximum(d.slip_max_abs),
        ))
    end

    df = DataFrame(summary_rows)
    CSV.write(joinpath(OUTDIR, "summary.csv"), df)

    df_sweep = df[df.mode .== "sweep", :]
    p_err = plot(df_sweep.noise_scale, df_sweep.pos_err, marker=:circle, linewidth=2,
        xlabel="noise scale", ylabel="pos_err", title="Open-loop pos error sweep (0.8~1.2)", label="pos_err")
    savefig(p_err, joinpath(OUTDIR, "sweep_pos_err.png"))

    p_gamma = plot(df_sweep.noise_scale, df_sweep.min_gamma_sum, marker=:circle, linewidth=2,
        xlabel="noise scale", ylabel="min gamma_sum", title="Open-loop min gamma_sum sweep", label="min gamma_sum")
    savefig(p_gamma, joinpath(OUTDIR, "sweep_min_gamma_sum.png"))

    df_pert = df[df.mode .== "perturb", :]
    p_pert = histogram(df_pert.pos_err, bins=min(10, nrow(df_pert)), xlabel="pos_err", ylabel="count",
        title="Perturb sensitivity at noise=1.0", label="")
    savefig(p_pert, joinpath(OUTDIR, "perturb_pos_err_hist.png"))

    println("saved: ", joinpath(OUTDIR, "summary.csv"))
    println("saved: ", joinpath(OUTDIR, "timeseries_open_noise_0.5.csv"))
    println("saved: ", joinpath(OUTDIR, "timeseries_open_noise_1.0.csv"))
    println("saved: ", joinpath(OUTDIR, "timeseries_open_noise_1.5.csv"))
    println("saved: ", joinpath(OUTDIR, "compare_gamma_sum_scales.png"))
    println("saved: ", joinpath(OUTDIR, "compare_phi_max_scales.png"))
    println("saved: ", joinpath(OUTDIR, "compare_slip_scales.png"))
    println("saved: ", joinpath(OUTDIR, "compare_theta_scales.png"))
    println("saved: ", joinpath(OUTDIR, "sweep_pos_err.png"))
    println("saved: ", joinpath(OUTDIR, "sweep_min_gamma_sum.png"))
    println("saved: ", joinpath(OUTDIR, "perturb_pos_err_hist.png"))
end

run()
