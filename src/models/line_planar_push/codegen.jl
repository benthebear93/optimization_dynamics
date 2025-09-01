path = @get_scratch!("lineplanarpush")

nq = lineplanarpush.nq
nu = lineplanarpush.nu
nc = lineplanarpush.nc
nz = num_var(lineplanarpush)
nθ = 2 * lineplanarpush.nq + lineplanarpush.nu + lineplanarpush.nw + 1 

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r_lpp = residual(lineplanarpush, z, θ, κ)
rz_lpp = Symbolics.jacobian(r_lpp, z)
rθ_lpp = Symbolics.jacobian(r_lpp, θ)

# Build function
r_lpp_func = build_function(r_lpp, z, θ, κ)[2]
rz_lpp_func = build_function(rz_lpp, z, θ)[2]
rθ_lpp_func = build_function(rθ_lpp, z, θ)[2]
rz_lpp_array = similar(rz_lpp, Float64)
rθ_lpp_array = similar(rθ_lpp, Float64)

@save joinpath(path, "residual.jld2") r_lpp_func rz_lpp_func rθ_lpp_func rz_lpp_array rθ_lpp_array
@load joinpath(path, "residual.jld2") r_lpp_func rz_lpp_func rθ_lpp_func rz_lpp_array rθ_lpp_array
