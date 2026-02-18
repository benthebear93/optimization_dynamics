function RoboDojo.indices_z(model::TipOverPush)
    nq = model.nq
    nc = model.nc
    nb = 2 * nc

    q = collect(1:nq)
    γ = collect(nq .+ (1:nc))
    b = collect(nq + nc .+ (1:nb))
    ψ = collect(nq + nc + nb .+ (1:nc))
    sγ = collect(nq + nc + nb + nc .+ (1:nc))
    sb = collect(nq + nc + nb + nc + nc .+ (1:nb))
    sψ = collect(nq + nc + nb + nc + nc + nb .+ (1:nc))

    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

function RoboDojo.nominal_configuration(model::TipOverPush)
    box_center_z = model.box_half_height + 1.0e-8
    [
        0.0,
        0.0,
        box_center_z,
        0.0,
        -model.box_half_width - model.pusher_radius - model.pusher_gap,
        0.0,
        box_center_z,
    ]
end

function RoboDojo.indices_optimization(model::TipOverPush)
    nq = model.nq
    nc = model.nc
    nb = 2 * nc
    nz = num_var(model)

    ort_1 = collect(nq .+ (1:(nc + nb + nc)))
    ort_2 = collect(nq + nc + nb + nc .+ (1:(nc + nb + nc)))

    equr = collect(1:(nq + nc + nb + nc))
    ortr = collect((nq + nc + nb + nc) .+ (1:(nc + nb + nc)))

    return IndicesOptimization(
        nz,
        nz,
        [ort_1, ort_2],
        [ort_1, ort_2],
        Vector{Vector{Vector{Int}}}(),
        Vector{Vector{Vector{Int}}}(),
        equr,
        ortr,
        Int[],
        Vector{Vector{Int}}(),
        ortr,
    )
end

function RoboDojo.initialize_z!(z, model::TipOverPush, idx::IndicesZ, q)
    eps = 3.16227766e-2
    z[idx.q] .= q
    z[idx.γ] .= eps
    z[idx.sγ] .= eps
    z[idx.ψ] .= eps
    z[idx.b] .= eps
    z[idx.sψ] .= eps
    z[idx.sb] .= eps
    return z
end

function RoboDojo.indices_θ(model::TipOverPush; nf=0)
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

RoboDojo.num_var(model::TipOverPush) = model.nq + model.nc + 2 * model.nc + model.nc + model.nc + 2 * model.nc + model.nc
RoboDojo.friction_coefficients(model::TipOverPush{T}) where T = T[]
