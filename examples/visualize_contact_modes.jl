using OptimizationDynamics
using Plots
using JLD2
using CSV
using DataFrames

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "planar_push_data"
mkpath(OUTPUT_DIR)
const CACHE_PATH = joinpath(OUTPUT_DIR, "stick_slip_mode_cache.jld2")

# Workflow:
# 1) Set REBUILD_CACHE=true once to recompute mode data from trajectories and save cache.
# 2) Set REBUILD_CACHE=false for fast plotting from cache only.
const REBUILD_CACHE = false

# Mode thresholds
const CONTACT_TOL = 5.0e-3
const VT_STICK_TOL = 2.0e-3

# Relative tangential speed in block local frame (point pusher).
function point_tangential_speed(qt, qtp1, h, rot2)
    b_t = qt[1:2]
    b_tp1 = qtp1[1:2]
    p_t = qt[4:5]
    p_tp1 = qtp1[4:5]

    Rt = rot2(qt[3])
    Rtp1 = rot2(qtp1[3])

    r_t = transpose(Rt) * (p_t - b_t)
    r_tp1 = transpose(Rtp1) * (p_tp1 - b_tp1)
    return (r_tp1[2] - r_t[2]) / h
end

# Relative tangential speeds in block local frame (line pusher: two points).
function line_tangential_speeds(qt, qtp1, h, rot2)
    b_t = qt[1:2]
    b_tp1 = qtp1[1:2]

    p1_t = qt[4:5]
    p1_tp1 = qtp1[4:5]
    p2_t = qt[6:7]
    p2_tp1 = qtp1[6:7]

    Rt = rot2(qt[3])
    Rtp1 = rot2(qtp1[3])

    r1_t = transpose(Rt) * (p1_t - b_t)
    r1_tp1 = transpose(Rtp1) * (p1_tp1 - b_tp1)
    r2_t = transpose(Rt) * (p2_t - b_t)
    r2_tp1 = transpose(Rtp1) * (p2_tp1 - b_tp1)

    vt1 = (r1_tp1[2] - r1_t[2]) / h
    vt2 = (r2_tp1[2] - r2_t[2]) / h
    return vt1, vt2
end

function rebuild_cache!()
    # Include point script and snapshot needed globals.
    include("point_push_free_box.jl")
    q_point = deepcopy(Main.q_sol)
    h_point = Main.h
    rot2_point = Main.rot2

    # Include line script and snapshot needed globals.
    include("line_push_free_box.jl")
    q_line = deepcopy(Main.q_sol)
    h_line = Main.h
    rot2_line = Main.rot2

    Np = length(q_point) - 1
    Nl = length(q_line) - 1
    N = min(Np, Nl)

    point_mode = zeros(Float64, N)
    line_mode = zeros(Float64, N)

    for t in 1:N
        # point mode: 0=no contact, 1=stick, 3=slip
        qp = q_point[t]
        qp1 = q_point[t + 1]
        phi_p = ϕ_func(planarpush, qp1)[1]
        in_contact_p = phi_p <= CONTACT_TOL
        vt_p = point_tangential_speed(qp, qp1, h_point, rot2_point)
        if !in_contact_p
            point_mode[t] = 0.0
        else
            point_mode[t] = abs(vt_p) <= VT_STICK_TOL ? 1.0 : 3.0
        end

        # line mode: 0=no contact, 1=both stick, 2=one-point slip, 3=two-point slip
        ql = q_line[t]
        ql1 = q_line[t + 1]
        phi_l = ϕ_func(lineplanarpush_xy, ql1)
        c1 = phi_l[1] <= CONTACT_TOL
        c2 = phi_l[2] <= CONTACT_TOL
        vt1, vt2 = line_tangential_speeds(ql, ql1, h_line, rot2_line)
        s1 = c1 && (abs(vt1) > VT_STICK_TOL)
        s2 = c2 && (abs(vt2) > VT_STICK_TOL)

        if !(c1 || c2)
            line_mode[t] = 0.0
        elseif s1 && s2
            line_mode[t] = 3.0
        elseif s1 || s2
            line_mode[t] = 2.0
        else
            line_mode[t] = 1.0
        end
    end

    h_cache = min(h_point, h_line)

    @save CACHE_PATH point_mode line_mode h_cache CONTACT_TOL VT_STICK_TOL
    println("cache saved: ", CACHE_PATH)
end

function plot_from_cache()
    if !isfile(CACHE_PATH)
        error("Cache not found: $CACHE_PATH. Set REBUILD_CACHE=true once and run again.")
    end

    point_mode = nothing
    line_mode = nothing
    h_cache = nothing
    @load CACHE_PATH point_mode line_mode h_cache

    N = min(length(point_mode), length(line_mode))
    point_mode = point_mode[1:N]
    line_mode = line_mode[1:N]

    # 2-band categorical strip image (thick rows for readability)
    band_h = 16
    gap_h = 4
    total_h = 2 * band_h + gap_h
    Z = zeros(Float64, total_h, N)
    Z[1:band_h, :] .= reshape(point_mode, 1, :)
    Z[(band_h + gap_h + 1):end, :] .= reshape(line_mode, 1, :)

    t = collect(0:h_cache:(N - 1) * h_cache)
    y_ticks = [band_h / 2, band_h + gap_h + band_h / 2]

    # Discrete palette: no-contact, stick, one-slip, two-slip
    c_no = RGB(0.06, 0.13, 0.45)
    c_st = RGB(0.12, 0.67, 0.62)
    c_1s = RGB(0.95, 0.62, 0.12)
    c_2s = RGB(0.98, 0.90, 0.12)
    cm = cgrad([c_no, c_st, c_1s, c_2s], [0.0, 1 / 3, 2 / 3, 1.0], categorical=true)

    p = heatmap(
        t,
        1:total_h,
        Z,
        c=cm,
        clim=(0, 3),
        xlabel="time (s)  |  each column = one timestep",
        ylabel="model",
        yticks=(y_ticks, ["point", "line"]),
        colorbar=false,
        size=(1100, 320),
        framestyle=:box,
    )

    title!(p, "Stick-Slip Mode")
    # Legend entries for categorical colors (dummy points outside visible area)
    scatter!(p, [t[1] - 1.0], [-1.0], markercolor=c_no, markersize=6, label="no contact")
    scatter!(p, [t[1] - 1.0], [-1.0], markercolor=c_st, markersize=6, label="stick")
    scatter!(p, [t[1] - 1.0], [-1.0], markercolor=c_1s, markersize=6, label="line: one-slip")
    scatter!(p, [t[1] - 1.0], [-1.0], markercolor=c_2s, markersize=6, label="slip")
    plot!(p, legend=:outertopright)
    out_path = joinpath(OUTPUT_DIR, "stick_slip_mode_strip_point_line.png")
    savefig(p, out_path)
    println("saved: ", out_path)

    # Save raw mode timeline for external plotting (MATLAB/Python/etc.)
    function mode_label_point(m)
        m == 0.0 ? "no_contact" : m == 1.0 ? "stick" : m == 3.0 ? "slip" : "unknown"
    end
    function mode_label_line(m)
        m == 0.0 ? "no_contact" : m == 1.0 ? "both_stick" : m == 2.0 ? "one_point_slip" : m == 3.0 ? "two_point_slip" : "unknown"
    end

    mode_df = DataFrame(
        t = t,
        point_mode_code = point_mode,
        point_mode_label = [mode_label_point(m) for m in point_mode],
        line_mode_code = line_mode,
        line_mode_label = [mode_label_line(m) for m in line_mode],
    )
    csv_path = joinpath(OUTPUT_DIR, "stick_slip_mode_timeline.csv")
    CSV.write(csv_path, mode_df)
    println("saved: ", csv_path)
end

if REBUILD_CACHE
    rebuild_cache!()
end
plot_from_cache()
