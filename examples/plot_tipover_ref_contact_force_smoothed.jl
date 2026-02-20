using JSON3
using Plots
using Statistics

const REF_JSON = joinpath(@__DIR__, "..", "tipover_ref_traj.json")
const OUT_DIR = joinpath(@__DIR__, "..", "data")
const OUT_PNG = joinpath(OUT_DIR, "tipover_ref_contact_force_smoothed.png")

# Strong smoothing defaults (plot-only):
const UPSAMPLE = 40
const GAUSS_RADIUS = 24
const GAUSS_SIGMA = 8.0

to_float_vec(x) = Float64[v for v in x]

function linear_resample(t_src::Vector{Float64}, y_src::Vector{Float64}, t_dst::Vector{Float64})
    n = length(t_src)
    n == length(y_src) || error("time/value length mismatch")
    n > 0 || return Float64[]
    n == 1 && return fill(y_src[1], length(t_dst))

    y_dst = Vector{Float64}(undef, length(t_dst))
    i = 1
    for k in eachindex(t_dst)
        t = t_dst[k]
        if t <= t_src[1]
            y_dst[k] = y_src[1]
            continue
        end
        if t >= t_src[end]
            y_dst[k] = y_src[end]
            continue
        end
        while i < n - 1 && t > t_src[i + 1]
            i += 1
        end
        t0 = t_src[i]
        t1 = t_src[i + 1]
        y0 = y_src[i]
        y1 = y_src[i + 1]
        α = (t - t0) / (t1 - t0 + 1.0e-12)
        y_dst[k] = (1.0 - α) * y0 + α * y1
    end
    return y_dst
end

function gaussian_kernel(radius::Int, sigma::Float64)
    radius >= 1 || error("radius must be >= 1")
    sigma > 0 || error("sigma must be > 0")
    xs = collect(-radius:radius)
    k = exp.(-0.5 .* (xs ./ sigma) .^ 2)
    k ./= sum(k)
    return k
end

function smooth_gaussian(y::Vector{Float64}, kernel::Vector{Float64})
    n = length(y)
    m = length(kernel)
    r = (m - 1) ÷ 2
    ys = similar(y)
    for i in 1:n
        acc = 0.0
        for j in 1:m
            idx = clamp(i + j - (r + 1), 1, n)
            acc += kernel[j] * y[idx]
        end
        ys[i] = acc
    end
    return ys
end

function main()
    isfile(REF_JSON) || error("missing reference json: " * REF_JSON)
    data = JSON3.read(read(REF_JSON, String))

    h = Float64(data["h"])
    gamma_raw = data["gamma"]
    H = length(gamma_raw)
    H > 1 || error("gamma horizon too short: H=$(H)")

    t = collect(0:(H - 1)) .* h
    nd = (H - 1) * UPSAMPLE + 1
    td = collect(LinRange(t[1], t[end], nd))
    kernel = gaussian_kernel(GAUSS_RADIUS, GAUSS_SIGMA)

    gammas = [to_float_vec(g) for g in gamma_raw]
    nc = length(gammas[1])
    nc >= 5 || error("expected at least 5 contact forces, got nc=$(nc)")

    g_raw = [Float64[g[i] for g in gammas] for i in 1:5]
    g_dense = [linear_resample(t, g_raw[i], td) for i in 1:5]
    g_smooth = [smooth_gaussian(g_dense[i], kernel) for i in 1:5]
    g_sum = [g_smooth[1][i] + g_smooth[2][i] + g_smooth[3][i] + g_smooth[4][i] + g_smooth[5][i] for i in eachindex(td)]

    p = plot(
        td,
        g_smooth[1],
        label="gamma1",
        xlabel="time [s]",
        ylabel="normal force",
        title="TipOver Contact Normal Forces (Smoothed from Reference)",
        linewidth=2.5,
        alpha=0.95,
    )
    plot!(p, td, g_smooth[2], label="gamma2", linewidth=2.5, alpha=0.95)
    plot!(p, td, g_smooth[3], label="gamma3", linewidth=2.5, alpha=0.95)
    plot!(p, td, g_smooth[4], label="gamma4", linewidth=2.5, alpha=0.95)
    plot!(p, td, g_smooth[5], label="gamma5 (pusher)", linewidth=2.5, alpha=0.95)
    plot!(p, td, g_sum, label="gamma_sum", linestyle=:dash, linewidth=3.0)

    mkpath(OUT_DIR)
    savefig(p, OUT_PNG)
    println("saved plot: ", OUT_PNG)
    println("h: ", h, ", H: ", H, ", duration: ", t[end], " s")
    println("upsample: ", UPSAMPLE, ", gaussian radius: ", GAUSS_RADIUS, ", sigma: ", GAUSS_SIGMA)
end

main()
