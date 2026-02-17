"""
    TipOverPush block
        2D tip-over dynamics in x-z plane with a dynamic pusher.
"""
struct TipOverPush{T} <: Model{T}
    # dimensions
    nq::Int
    nu::Int
    nw::Int
    nc::Int
    nc_impact::Int

    mb::T
    mp::T
    inertia::T
    gravity::T
    μ_floor::T
    μ_pusher::T

    box_half_width::T
    box_half_height::T
    pusher_radius::T
    pusher_gap::T
    pusher_height::T
end

function rotation_matrix_pitch(θ)
    c = cos(θ)
    s = sin(θ)
    [
        c 0.0 s
        0.0 1.0 0.0
        -s 0.0 c
    ]
end

M_func(model::TipOverPush, q) = Diagonal([model.mb, model.mb, model.mb, model.inertia, model.mp, model.mp, model.mp])

function C_func(model::TipOverPush, q, q̇)
    [0.0, 0.0, model.mb * model.gravity, 0.0, 0.0, 0.0, 0.0]
end

function B_func(model::TipOverPush, q)
    [
        0.0 0.0
        0.0 0.0
        0.0 0.0
        0.0 0.0
        1.0 0.0
        0.0 0.0
        0.0 1.0
    ]
end

function floor_corner_positions(model::TipOverPush, q)
    box_pos = q[1:3]
    θ = q[4]
    R = rotation_matrix_pitch(θ)

    c1 = box_pos + R * [model.box_half_width, 0.0, -model.box_half_height]
    c2 = box_pos + R * [-model.box_half_width, 0.0, -model.box_half_height]
    c3 = box_pos + R * [model.box_half_width, 0.0, model.box_half_height]
    c4 = box_pos + R * [-model.box_half_width, 0.0, model.box_half_height]
    return c1, c2, c3, c4
end

function ϕ_func(model::TipOverPush, q)
    c1, c2, c3, c4 = floor_corner_positions(model, q)

    box_pos = q[1:3]
    pusher_pos = q[5:7]
    R = rotation_matrix_pitch(q[4])

    δ = pusher_pos - box_pos
    p_local = transpose(R) * δ

    a = model.box_half_width + model.pusher_radius
    b = model.box_half_height + model.pusher_radius
    s = 10
    nx = p_local[1] / a
    nz = p_local[3] / b
    ϕ_push = (nx^s + nz^s + 1.0e-12)^(1.0 / s) - 1.0

    [c1[3], c2[3], c3[3], c4[3], ϕ_push]
end

function N_func(model::TipOverPush, q)
    ϕ = ϕ_func(model, q)
    Symbolics.jacobian(ϕ, q)
end

function P_func(model::TipOverPush, q)
    θ = q[4]
    R = rotation_matrix_pitch(θ)

    # Four floor contacts, two friction rays (+x and -x) per contact.
    r1 = R * [model.box_half_width, 0.0, -model.box_half_height]
    r2 = R * [-model.box_half_width, 0.0, -model.box_half_height]
    r3 = R * [model.box_half_width, 0.0, model.box_half_height]
    r4 = R * [-model.box_half_width, 0.0, model.box_half_height]

    p11 = [1.0, 0.0, 0.0, r1[3], 0.0, 0.0, 0.0]
    p12 = [-1.0, 0.0, 0.0, -r1[3], 0.0, 0.0, 0.0]

    p21 = [1.0, 0.0, 0.0, r2[3], 0.0, 0.0, 0.0]
    p22 = [-1.0, 0.0, 0.0, -r2[3], 0.0, 0.0, 0.0]

    p31 = [1.0, 0.0, 0.0, r3[3], 0.0, 0.0, 0.0]
    p32 = [-1.0, 0.0, 0.0, -r3[3], 0.0, 0.0, 0.0]

    p41 = [1.0, 0.0, 0.0, r4[3], 0.0, 0.0, 0.0]
    p42 = [-1.0, 0.0, 0.0, -r4[3], 0.0, 0.0, 0.0]

    # Pusher contact tangent from signed-distance normal.
    N = N_func(model, q)
    nx = N[5, 5]
    nz = N[5, 7]
    n_norm = sqrt(nx^2 + nz^2 + 1.0e-12)
    tx0 = -nz / n_norm
    tz0 = nx / n_norm

    rbx = q[5] - q[1]
    rbz = q[7] - q[3]

    τp1 = -rbz * tx0 + rbx * tz0
    τp2 = -τp1
    p51 = [-tx0, 0.0, -tz0, τp1, tx0, 0.0, tz0]
    p52 = [tx0, 0.0, tz0, τp2, -tx0, 0.0, -tz0]

    [
        p11'
        p12'
        p21'
        p22'
        p31'
        p32'
        p41'
        p42'
        p51'
        p52'
    ]
end

function residual(model::TipOverPush, z, θ, κ)
    nq = model.nq
    nu = model.nu
    nw = model.nw
    nc = model.nc
    nb = 2 * nc

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    w1 = θ[2nq + nu .+ (1:nw)]
    h = θ[2nq + nu + nw .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc)]
    b1 = z[nq + nc .+ (1:nb)]
    ψ1 = z[nq + nc + nb .+ (1:nc)]
    sγ1 = z[nq + nc + nb + nc .+ (1:nc)]
    sb1 = z[nq + nc + nb + nc + nc .+ (1:nb)]
    sψ1 = z[nq + nc + nb + nc + nc + nb .+ (1:nc)]

    ϕ = ϕ_func(model, q2)
    N = N_func(model, q2)
    P = P_func(model, q2)

    vT = P * ((q2 - q1) ./ h[1])
    ψ_stack = transpose([
        1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0
    ]) * ψ1

    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm2, vm2)

    dyn = 0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2 + B_func(model, qm2) * u1 + transpose(N) * γ1 + transpose(P) * b1

    μ_contact = [model.μ_floor, model.μ_floor, model.μ_floor, model.μ_floor, model.μ_pusher]
    E = [
        1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 1.0
    ]

    res_sd = ϕ - sγ1
    res_vt = sb1 - vT - ψ_stack
    res_fric = sψ1 - (μ_contact .* γ1 - E * b1)

    [
        dyn
        res_sd
        res_vt
        res_fric
        γ1 .* sγ1 .- κ[1]
        b1 .* sb1 .- κ[1]
        ψ1 .* sψ1 .- κ[1]
    ]
end

# dimensions / parameters
nq = 7
nu = 2
nw = 0
nc = 5
nc_impact = 5

box_half_width = 0.09
box_half_height = 0.11
pusher_radius = 0.02
pusher_gap = 0.004
pusher_height = 0.17
mb = 1.2
mp = 10.0
gravity = 9.81
μ_floor = 0.6
μ_pusher = 0.5
inertia = (1.0 / 12.0) * mb * ((2.0 * box_half_width)^2 + (2.0 * box_half_height)^2)

tipoverpush = TipOverPush(
    nq,
    nu,
    nw,
    nc,
    nc_impact,
    mb,
    mp,
    inertia,
    gravity,
    μ_floor,
    μ_pusher,
    box_half_width,
    box_half_height,
    pusher_radius,
    pusher_gap,
    pusher_height,
)
