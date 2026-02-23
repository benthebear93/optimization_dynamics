using JSON3
using Plots

const REF_JSON = joinpath(@__DIR__, "..", "data", "reference_trajectory", "pusher_ref_traj_complex.json")
const OUT_DIR = joinpath(@__DIR__, "..", "data")
const OUT_PNG = joinpath(OUT_DIR, "planar_push_complex_pusher_box_force.png")

to_float_vec(x) = Float64[v for v in x]

function main()
    isfile(REF_JSON) || error("missing reference json: " * REF_JSON)
    data = JSON3.read(read(REF_JSON, String))

    h = Float64(data["h"])
    gamma_raw = data["gamma"]
    H = length(gamma_raw)
    H > 0 || error("gamma horizon too short: H=$(H)")

    gammas = [to_float_vec(g) for g in gamma_raw]
    nc = length(gammas[1])
    nc > 0 || error("gamma dimension is zero")

    # In this planar push setup, pusher-box normal force is stored in the last gamma component.
    pusher_idx = nc
    force = [gammas[t][pusher_idx] for t in 1:H]
    t = collect(0:(H - 1)) .* h

    p = plot(
        t,
        force;
        xlabel="time [s]",
        ylabel="force",
        title="Planar Push Complex Pusher-Box Contact Force",
        label="gamma$(pusher_idx) (pusher-box)",
        linewidth=2.5,
        alpha=0.95,
    )

    mkpath(OUT_DIR)
    savefig(p, OUT_PNG)
    println("saved plot: ", OUT_PNG)
    println("h: ", h, ", H: ", H, ", duration: ", t[end], " s")
    println("nc: ", nc, ", selected gamma index: ", pusher_idx)
end

main()
