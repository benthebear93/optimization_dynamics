function RoboDojo.indices_z(model::LinePlanarPushXY)
    nq = model.nq
    q = collect(1:nq)
    γ = collect(nq .+ (1:2))
    sγ = collect(nq + 2 .+ (1:2))
    ψ = collect(nq + 2 + 2 .+ (1:6))
    b = collect(nq + 2 + 2 + 6 .+ (1:10))
    sψ = collect(nq + 2 + 2 + 6 + 10 .+ (1:6))
    sb = collect(nq + 2 + 2 + 6 + 10 + 6 .+ (1:10))
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.nominal_configuration(model::LinePlanarPushXY) = [0.0; 0.0; 0.0; -0.1 - 1.0e-8; 0.025; -0.1 - 1.0e-8; -0.025]

function RoboDojo.indices_optimization(model::LinePlanarPushXY)
    nq = model.nq
    nz = num_var(model)

    ort1 = [nq + 1, nq + 2]
    ort2 = [nq + 3, nq + 4]

    socz = Vector{Vector{Vector{Int}}}([
        [[nq + 5, nq + 11, nq + 12], [nq + 21, nq + 27, nq + 28]],
        [[nq + 6, nq + 13, nq + 14], [nq + 22, nq + 29, nq + 30]],
        [[nq + 7, nq + 15, nq + 16], [nq + 23, nq + 31, nq + 32]],
        [[nq + 8, nq + 17, nq + 18], [nq + 24, nq + 33, nq + 34]],
        [[nq + 9, nq + 19],          [nq + 25, nq + 35]],
        [[nq + 10, nq + 20],         [nq + 26, nq + 36]],
    ])
    socΔ = socz

    equr = collect(1:(nq + 18))
    ortr = [nq + 19, nq + 20]
    socr = collect((nq + 20) .+ (1:16))
    socri = Vector{Vector{Int}}([
        collect((nq + 20) .+ (1:3)),
        collect((nq + 23) .+ (1:3)),
        collect((nq + 26) .+ (1:3)),
        collect((nq + 29) .+ (1:3)),
        collect((nq + 32) .+ (1:2)),
        collect((nq + 34) .+ (1:2)),
    ])
    bil = collect((nq + 18) .+ (1:18))

    return IndicesOptimization(
        nz, nz,
        [ort1, ort2],
        [ort1, ort2],
        socz, socΔ,
        equr, ortr, socr, socri, bil
    )
end

function RoboDojo.initialize_z!(z, model::LinePlanarPushXY, idx::IndicesZ, q)
    z[idx.q] .= q
	z[idx.γ] .= 1.0
	z[idx.sγ] .= 1.0
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end

function RoboDojo.indices_θ(model::LinePlanarPushXY; nf=0)
    nq = model.nq
    nu = model.nu
    nw = model.nw

    q1 = collect(1:nq)
    q2 = collect(nq .+ (1:nq))
    u = collect(2nq .+ (1:nu))
    w = collect(2nq + nu .+ (1:nw))
    f = collect(2nq + nu + nw .+ (1:nf))
    h = collect(2nq + nu + nw + nf .+ (1:1))

    Indicesθ(q1, q2, u, w, f, h)
end

RoboDojo.num_var(model::LinePlanarPushXY) = 43
friction_coefficients(model::LinePlanarPushXY{T}) where T = T[]
