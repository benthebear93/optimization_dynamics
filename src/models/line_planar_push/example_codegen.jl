using OptimizationDynamics
using Scratch 
using Symbolics 
using LinearAlgebra
using JLD2
include("model.jl")

function num_var(model)
    # nq = model.nq
    # nc = model.nc
    # nb = 2
    q = 5  # 3
    gamma = 2  # 1
    s_gamma = 2  # 1
    psi = 6  # 5
    b = 10  # 9
    spsi = 6  # 5
    sb = 10  # 9
    return q + gamma + s_gamma + psi + b + spsi + sb
end

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
rz_lpp = Symbolics.jacobian(r_pp, z)
rθ_lpp = Symbolics.jacobian(r_pp, θ)

# Build function
r_lpp_func = build_function(r_lpp, z, θ, κ)[2]
rz_lpp_func = build_function(rz_lpp, z, θ)[2]
rθ_lpp_func = build_function(rθ_lpp, z, θ)[2]
rz_lpp_array = similar(rz_lpp, Float64)
rθ_lpp_array = similar(rθ_lpp, Float64)

@save joinpath(path, "residual.jld2") r_lpp_func rz_lpp_func rθ_lpp_func rz_lpp_array rθ_lpp_array
@load joinpath(path, "residual.jld2") r_lpp_func rz_lpp_func rθ_lpp_func rz_lpp_array rθ_lpp_array