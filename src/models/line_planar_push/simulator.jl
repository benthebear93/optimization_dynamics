function RoboDojo.indices_z(model::LinePlanarPush) 
    nq = model.nq 
    nc = model.nc
    q = collect(1:nq) 
    γ = collect(nq .+ (1:1)) 
    sγ = collect(nq + 1 .+ (1:1))
    ψ = collect(nq + 2 .+ (1:6)) 
    b = collect(nq + 2 + 6 .+ (1:10)) 
    sψ = collect(nq + 2 + 6 + 10 .+ (1:6)) 
    sb = collect(nq + 2 + 6 + 10 + 6 .+ (1:10)) 
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.nominal_configuration(model::LinePlanarPush) = [0.0; 0.0; 0.0; 0.0; 0.0]

function RoboDojo.indices_optimization(model::LinePlanarPush)
    nq = model.nq
    nz = num_var(model)
    println("index nz", nz)
    nc = 2
    # ORThant (γ, s_γ): Python [nq], [nq+1]  -> Julia [nq+1], [nq+2]
    ort1 = [nq + 1, nq + 2]
    ort2 = [nq + 2, nq + 2]

    # SOC blocks (ψ_i, b-slices, s_ψ_i, s_b-slices)
    # Python 예시를 +1 보정해 그대로 옮김
    socz = Vector{Vector{Vector{Int}}}([
        [[nq + 5, nq + 11, nq + 12], [nq + 21, nq + 27, nq + 28]],
        [[nq + 6, nq + 13, nq + 14], [nq + 22, nq + 29, nq + 30]],
        [[nq + 7, nq + 15, nq + 16], [nq + 23, nq + 31, nq + 32]],
        [[nq + 8, nq + 17, nq + 18], [nq + 24, nq + 33, nq + 34]],
        [[nq + 9, nq + 19],          [nq + 25, nq + 35]],
        [[nq + 10,nq + 20],          [nq + 26, nq + 36]],
    ])
    socΔ = socz  # Δz도 동일 블록 구성

    # Residual indices
    # equr: Python 0:(nq+14) -> Julia 1:(nq+15)
    equr = collect(1:(nq + 18))
    # ortr: Python [nq+15] -> Julia [nq+16]
    ortr = [nq + 19, nq + 20]
    # socr: Python (nq+16):(nq+29) -> Julia (nq+17):(nq+30)
    socr = collect((nq + 19) .+ (1:16))
    # socri: Python 블록들을 +1 보정
    socri = Vector{Vector{Int}}([
        collect((nq + 19) .+ (1:3)),  # nq+17:nq+19
        collect((nq + 22) .+ (1:3)),  # nq+20:nq+22
        collect((nq + 25) .+ (1:3)),  # nq+23:nq+25
        collect((nq + 28) .+ (1:3)),  # nq+26:nq+28
        collect((nq + 31) .+ (1:2)),  # nq+29:nq+30
        collect((nq + 33) .+ (1:2)),  # nq+29:nq+30
    ])
    # bil: Python (nq+15):(nq+29) -> Julia (nq+16):(nq+30)
    bil = collect((nq + 18) .+ (1:18))

    return IndicesOptimization(
        nz, nz,
        [ort1, ort2],
        [ort1, ort2],
        socz, socΔ,
        equr, ortr, socr, socri, bil
    )
end

function RoboDojo.initialize_z!(z, model::LinePlanarPush, idx::IndicesZ, q)
    z[idx.q] .= q 
	z[idx.γ] .= 1.0 
	z[idx.sγ] .= 1.0
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end

function RoboDojo.num_var(model)
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

num_var(model::LinePlanarPush) = 41 # model.nq + 2 + 2 + 6 + 10 + 6 + 10
friction_coefficients(model::LinePlanarPush{T}) where T = T[]  


