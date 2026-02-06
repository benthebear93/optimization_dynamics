path = @get_scratch!("lineplanarpush_xy")

nq = lineplanarpush_xy.nq
nu = lineplanarpush_xy.nu
nc = lineplanarpush_xy.nc
nz = num_var(lineplanarpush_xy)
nθ = 2 * lineplanarpush_xy.nq + lineplanarpush_xy.nu + lineplanarpush_xy.nw + 1

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r_lppxy = residual(lineplanarpush_xy, z, θ, κ)
@show r_lppxy
rz_lppxy = Symbolics.jacobian(r_lppxy, z)
rθ_lppxy = Symbolics.jacobian(r_lppxy, θ)

# Build function
r_lppxy_func = build_function(r_lppxy, z, θ, κ)[2]
rz_lppxy_func = build_function(rz_lppxy, z, θ)[2]
rθ_lppxy_func = build_function(rθ_lppxy, z, θ)[2]
rz_lppxy_array = similar(rz_lppxy, Float64)
rθ_lppxy_array = similar(rθ_lppxy, Float64)

@save joinpath(path, "residual.jld2") r_lppxy_func rz_lppxy_func rθ_lppxy_func rz_lppxy_array rθ_lppxy_array
@load joinpath(path, "residual.jld2") r_lppxy_func rz_lppxy_func rθ_lppxy_func rz_lppxy_array rθ_lppxy_array
