using LinearAlgebra
using Plots

ENV["GKSwstype"] = "100"
const OUTPUT_DIR = "planar_push_data"
mkpath(OUTPUT_DIR)

module PointRun
include("point_push_free_box.jl")
end

module LineRun
include("line_push_free_box.jl")
end

gamma_comp(γ, i) = (γ isa AbstractVector && length(γ) >= i) ? γ[i] : 0.0

function point_metrics()
    q = PointRun.q_sol
    u = PointRun.u_sol
    γ = PointRun.gamma_sol
    h = PointRun.h

    box_goal = PointRun.qT[1:2]
    box_final = q[end][1:2]
    box_pos_err = box_final - box_goal
    θ_goal = PointRun.qT[3]
    θ_final = q[end][3]
    θ_err = θ_final - θ_goal

    control_effort = sum(dot(ui, ui) for ui in u)
    total_u_mag = sum(norm(ui) for ui in u)
    u_norm = [norm(ui) for ui in u]
    u1 = [ui[1] for ui in u]
    u2 = [ui[2] for ui in u]
    gamma_vals = [gamma_comp(gi, 1) for gi in γ]
    gamma_mean_abs = sum(abs.(gamma_vals)) / max(length(gamma_vals), 1)
    gamma_peak = maximum(abs.(gamma_vals))

    slip = [(transpose(PointRun.rot2(qt[3])) * (qt[4:5] - qt[1:2]))[2] - PointRun.pusher_y_ref for qt in q]
    slip_max_abs = maximum(abs.(slip))

    return (
        h=h,
        theta=[qt[3] for qt in q],
        slip=slip,
        u_norm=u_norm,
        u1=u1,
        u2=u2,
        box_pos_err=box_pos_err,
        theta_err=θ_err,
        control_effort=control_effort,
        total_u_mag=total_u_mag,
        gamma_mean_abs=gamma_mean_abs,
        gamma_peak=gamma_peak,
        slip_max_abs=slip_max_abs,
        slip_bound=PointRun.max_tangent_slip,
    )
end

function line_metrics()
    q = LineRun.q_sol
    u = LineRun.u_sol
    γ = LineRun.gamma_sol
    h = LineRun.h

    box_goal = LineRun.qT[1:2]
    box_final = q[end][1:2]
    box_pos_err = box_final - box_goal
    θ_goal = LineRun.qT[3]
    θ_final = q[end][3]
    θ_err = θ_final - θ_goal

    control_effort = sum(dot(ui, ui) for ui in u)
    total_u_mag = sum(norm(ui) for ui in u)
    u_norm = [norm(ui) for ui in u]
    u1 = [ui[1] for ui in u]
    u2 = [ui[2] for ui in u]
    u3 = [ui[3] for ui in u]
    u4 = [ui[4] for ui in u]
    gamma1 = [gamma_comp(gi, 1) for gi in γ]
    gamma2 = [gamma_comp(gi, 2) for gi in γ]
    gamma_mean_abs = (sum(abs.(gamma1)) + sum(abs.(gamma2))) / max(length(gamma1) + length(gamma2), 1)
    gamma_peak = maximum(abs.([gamma1; gamma2]))

    slip1 = Float64[]
    slip2 = Float64[]
    for t in 1:length(u)
        qt = q[t + 1]
        p_block = qt[1:2]
        Rwb = LineRun.rot2(qt[3])
        p1_local = transpose(Rwb) * (qt[4:5] - p_block)
        p2_local = transpose(Rwb) * (qt[6:7] - p_block)
        push!(slip1, p1_local[2] - LineRun.pusher_y_offset)
        push!(slip2, p2_local[2] + LineRun.pusher_y_offset)
    end

    slip_max_abs = max(maximum(abs.(slip1)), maximum(abs.(slip2)))

    return (
        h=h,
        theta=[qt[3] for qt in q],
        slip1=slip1,
        slip2=slip2,
        u_norm=u_norm,
        u1=u1,
        u2=u2,
        u3=u3,
        u4=u4,
        box_pos_err=box_pos_err,
        theta_err=θ_err,
        control_effort=control_effort,
        total_u_mag=total_u_mag,
        gamma_mean_abs=gamma_mean_abs,
        gamma_peak=gamma_peak,
        slip_max_abs=slip_max_abs,
        slip_bound=LineRun.max_tangent_slip,
    )
end

pm = point_metrics()
lm = line_metrics()

println("=== Point vs Line Summary ===")
println("point box_pos_err_norm = ", norm(pm.box_pos_err))
println("line  box_pos_err_norm = ", norm(lm.box_pos_err))
println("point theta_err_abs    = ", abs(pm.theta_err))
println("line  theta_err_abs    = ", abs(lm.theta_err))
println("point control_effort   = ", pm.control_effort)
println("line  control_effort   = ", lm.control_effort)
println("point total_u_mag      = ", pm.total_u_mag)
println("line  total_u_mag      = ", lm.total_u_mag)
println("point gamma_mean_abs   = ", pm.gamma_mean_abs, ", gamma_peak = ", pm.gamma_peak)
println("line  gamma_mean_abs   = ", lm.gamma_mean_abs, ", gamma_peak = ", lm.gamma_peak)
println("point slip_max_abs     = ", pm.slip_max_abs, " / bound ", pm.slip_bound)
println("line  slip_max_abs     = ", lm.slip_max_abs, " / bound ", lm.slip_bound)

# theta comparison
t_p = collect(0:pm.h:(length(pm.theta) - 1) * pm.h)
t_l = collect(0:lm.h:(length(lm.theta) - 1) * lm.h)
p_theta = plot(t_p, pm.theta, label="point theta", linewidth=2, color=:blue)
plot!(p_theta, t_l, lm.theta, label="line theta", linewidth=2, color=:red)
plot!(p_theta, t_p, fill(PointRun.qT[3], length(t_p)), label="theta goal", linewidth=2, color=:black, linestyle=:dash)

# control norm comparison
t_pu = collect(0:pm.h:(length(pm.u_norm) - 1) * pm.h)
t_lu = collect(0:lm.h:(length(lm.u_norm) - 1) * lm.h)
p_unorm = plot(t_pu, pm.u_norm, label="point ||u||", linewidth=2, color=:blue)
plot!(p_unorm, t_lu, lm.u_norm, label="line ||u||", linewidth=2, color=:red)

# control channels comparison
p_u = plot(t_pu, pm.u1, label="point u1", linewidth=2, color=:blue)
plot!(p_u, t_pu, pm.u2, label="point u2", linewidth=2, color=:cyan)
plot!(p_u, t_lu, lm.u1, label="line u1", linewidth=2, color=:red)
plot!(p_u, t_lu, lm.u2, label="line u2", linewidth=2, color=:orange)
plot!(p_u, t_lu, lm.u3, label="line u3", linewidth=2, color=:magenta)
plot!(p_u, t_lu, lm.u4, label="line u4", linewidth=2, color=:green)

# cumulative/summed control magnitude over horizon
pm_u_cumsum = cumsum(pm.u_norm)
lm_u_cumsum = cumsum(lm.u_norm)
t_pcum = collect(0:pm.h:(length(pm_u_cumsum) - 1) * pm.h)
t_lcum = collect(0:lm.h:(length(lm_u_cumsum) - 1) * lm.h)
p_ucum = plot(t_pcum, pm_u_cumsum, label="point cumulative sum ||u||", linewidth=2, color=:blue)
plot!(p_ucum, t_lcum, lm_u_cumsum, label="line cumulative sum ||u||", linewidth=2, color=:red)

# final total sum comparison (scalar)
p_utotal = bar(["point", "line"], [pm.total_u_mag, lm.total_u_mag], label="sum_t ||u_t||")

# slip comparison
t_pslip = collect(0:pm.h:(length(pm.slip) - 1) * pm.h)
t_lslip = collect(0:lm.h:(length(lm.slip1) - 1) * lm.h)
p_slip = plot(t_pslip, pm.slip, label="point slip", linewidth=2, color=:blue)
plot!(p_slip, t_pslip, fill(pm.slip_bound, length(t_pslip)), label="point slip bound", linewidth=1, color=:blue, linestyle=:dash)
plot!(p_slip, t_pslip, fill(-pm.slip_bound, length(t_pslip)), label="", linewidth=1, color=:blue, linestyle=:dash)
plot!(p_slip, t_lslip, lm.slip1, label="line slip1", linewidth=2, color=:red)
plot!(p_slip, t_lslip, lm.slip2, label="line slip2", linewidth=2, color=:orange)
plot!(p_slip, t_lslip, fill(lm.slip_bound, length(t_lslip)), label="line slip bound", linewidth=1, color=:red, linestyle=:dash)
plot!(p_slip, t_lslip, fill(-lm.slip_bound, length(t_lslip)), label="", linewidth=1, color=:red, linestyle=:dash)

savefig(p_theta, joinpath(OUTPUT_DIR, "compare_point_line_theta.png"))
savefig(p_unorm, joinpath(OUTPUT_DIR, "compare_point_line_u_norm.png"))
savefig(p_u, joinpath(OUTPUT_DIR, "compare_point_line_u_channels.png"))
savefig(p_ucum, joinpath(OUTPUT_DIR, "compare_point_line_u_cumsum.png"))
savefig(p_utotal, joinpath(OUTPUT_DIR, "compare_point_line_u_total_bar.png"))
savefig(p_slip, joinpath(OUTPUT_DIR, "compare_point_line_slip.png"))

println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_theta.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_u_norm.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_u_channels.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_u_cumsum.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_u_total_bar.png"))
println("saved: ", joinpath(OUTPUT_DIR, "compare_point_line_slip.png"))
