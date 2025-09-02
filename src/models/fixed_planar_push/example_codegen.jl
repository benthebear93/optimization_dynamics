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
    q = 3  # 3
    gamma = 1  # 1
    s_gamma = 1  # 1
    psi = 5  # 5
    b = 9  # 9
    spsi = 5  # 5
    sb = 9  # 9
    return q + gamma + s_gamma + psi + b + spsi + sb
end

path = @get_scratch!("fixedplanarpush")
nq = fixedplanarpush.nq
nu = fixedplanarpush.nu
nc = fixedplanarpush.nc
nz = num_var(fixedplanarpush)
nθ = 2 * fixedplanarpush.nq + fixedplanarpush.nu + fixedplanarpush.nw + 1 

# Declare variables
@variables z[1:nz]
@variables θ[1:nθ]
@variables κ[1:1]

# Residual
r_fpp = residual(fixedplanarpush, z, θ, κ)
rz_fpp = Symbolics.jacobian(r_fpp, z)
rθ_fpp = Symbolics.jacobian(r_fpp, θ)

# Build function
r_fpp_func = build_function(r_fpp, z, θ, κ)[2]
rz_fpp_func = build_function(rz_fpp, z, θ)[2]
rθ_fpp_func = build_function(rθ_fpp, z, θ)[2]
rz_fpp_array = similar(rz_fpp, Float64)
rθ_fpp_array = similar(rθ_fpp, Float64)

@save joinpath(path, "residual.jld2") r_fpp_func rz_fpp_func rθ_fpp_func rz_fpp_array rθ_fpp_array
@load joinpath(path, "residual.jld2") r_fpp_func rz_fpp_func rθ_fpp_func rz_fpp_array rθ_fpp_array