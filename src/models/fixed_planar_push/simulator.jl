function RoboDojo.indices_z(model::FixedPlanarPush) 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(nq .+ (1:1)) 
    sγ = collect(nq + 1 .+ (1:1))
    ψ = collect(nq + 2 .+ (1:5)) 
    b = collect(nq + 2 + 5 .+ (1:9)) 
    sψ = collect(nq + 2 + 5 + 9 .+ (1:5)) 
    sb = collect(nq + 2 + 5 + 9 + 5 .+ (1:9)) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.nominal_configuration(model::FixedPlanarPush) = [0.0; 0.0; 0.0]

function RoboDojo.indices_optimization(model::FixedPlanarPush)
    nq = model.nq
    nz = num_var(model)

    # ORThant (γ, s_γ): Python [nq], [nq+1]  -> Julia [nq+1], [nq+2]
    ort1 = [nq + 1]
    ort2 = [nq + 2]

    # SOC blocks (ψ_i, b-slices, s_ψ_i, s_b-slices)
    # Python 예시를 +1 보정해 그대로 옮김
    socz = Vector{Vector{Vector{Int}}}([
        [[nq + 3, nq + 8,  nq + 9],  [nq + 17, nq + 22, nq + 23]],
        [[nq + 4, nq + 10, nq + 11], [nq + 18, nq + 24, nq + 25]],
        [[nq + 5, nq + 12, nq + 13], [nq + 19, nq + 26, nq + 27]],
        [[nq + 6, nq + 14, nq + 15], [nq + 20, nq + 28, nq + 29]],
        [[nq + 7, nq + 16],          [nq + 21, nq + 30]],
    ])
    socΔ = socz  # Δz도 동일 블록 구성

    # Residual indices
    # equr: Python 0:(nq+14) -> Julia 1:(nq+15)
    equr = collect(1:(nq + 15))
    # ortr: Python [nq+15] -> Julia [nq+16]
    ortr = [nq + 16]
    # socr: Python (nq+16):(nq+29) -> Julia (nq+17):(nq+30)
    socr = collect((nq + 16) .+ (1:14))
    # socri: Python 블록들을 +1 보정
    socri = Vector{Vector{Int}}([
        collect((nq + 16) .+ (1:3)),  # nq+17:nq+19
        collect((nq + 19) .+ (1:3)),  # nq+20:nq+22
        collect((nq + 22) .+ (1:3)),  # nq+23:nq+25
        collect((nq + 25) .+ (1:3)),  # nq+26:nq+28
        collect((nq + 28) .+ (1:2)),  # nq+29:nq+30
    ])
    # bil: Python (nq+15):(nq+29) -> Julia (nq+16):(nq+30)
    bil = collect((nq + 15) .+ (1:15))

    return IndicesOptimization(
        nz, nz,
        [ort1, ort2],
        [ort1, ort2],
        socz, socΔ,
        equr, ortr, socr, socri, bil
    )
end

function RoboDojo.initialize_z!(z, model::FixedPlanarPush, idx::IndicesZ, q)
    z[idx.q] .= q 
	z[idx.γ] .= 1.0 
	z[idx.sγ] .= 1.0
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end

# function num_var(model)
#     # nq = model.nq
#     # nc = model.nc
#     # nb = 2
#     q = 3  # 3
#     gamma = 1  # 1
#     s_gamma = 1  # 1
#     psi = 5  # 5
#     b = 9  # 9
#     spsi = 5  # 5
#     sb = 9  # 9
#     return q + gamma + s_gamma + psi + b + spsi + sb
# end

num_var(model::FixedPlanarPush) = model.nq + 1 + 1 + 5 + 9 + 5 + 9
friction_coefficients(model::FixedPlanarPush{T}) where T = T[]  


