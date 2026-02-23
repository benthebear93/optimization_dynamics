import OptimizationDynamics: ϕ_func

"""
    planar push block (v2)
        parameter-aligned variant for Python pusher settings.
"""
struct PlanarPushV2{T} <: Model{T}
    nq::Int
    nu::Int
    nw::Int
    nc::Int
    nc_impact::Int
    mass_block::T
    mass_pusher::T
    inertia::T
    mu_body::T
    mu_pusher::T
    gravity::T
    r_box::T
    r_pusher::T
    contact_corner_offset::Vector{Vector{T}}
end

function rotation_matrix_v2(x)
    [cos(x) -sin(x); sin(x) cos(x)]
end

function sd_2d_box_v2(p, pose, r_box, r_pusher)
    x, y, θ = pose
    Δ = rotation_matrix_v2(-θ) * (p - [x, y])
    s = 10
    sum(Δ.^s)^(1 / s) - (r_box + r_pusher)
end

M_func(model::PlanarPushV2, q) = Diagonal([
    model.mass_block,
    model.mass_block,
    model.inertia,
    model.mass_pusher,
    model.mass_pusher,
])

function C_func(model::PlanarPushV2, q, q̇)
    [0.0, 0.0, 0.0, 0.0, 0.0]
end

function ϕ_func(model::PlanarPushV2, q)
    p_block = q[1:3]
    p_pusher = q[4:5]
    sdf = sd_2d_box_v2(p_pusher, p_block, model.r_box, model.r_pusher)
    [sdf]
end

function B_func(model::PlanarPushV2, q)
    [0.0 0.0;
     0.0 0.0;
     0.0 0.0;
     1.0 0.0;
     0.0 1.0]
end

function N_func(model::PlanarPushV2, q)
    ϕ = ϕ_func(model, q)
    vec(Symbolics.jacobian(ϕ, q))
end

function p_func(model::PlanarPushV2, x)
    pos = x[1:2]
    θ = x[3]
    R = rotation_matrix_v2(θ)

    [(pos + R * model.contact_corner_offset[1])[1:2];
     (pos + R * model.contact_corner_offset[2])[1:2];
     (pos + R * model.contact_corner_offset[3])[1:2];
     (pos + R * model.contact_corner_offset[4])[1:2]]
end

function P_func(model::PlanarPushV2, q)
    pf = p_func(model, q)
    P_block = Symbolics.jacobian(pf, q)

    p_block = q[1:3]
    p_pusher = q[4:5]

    ϕ = ϕ_func(model, q)
    N = vec(Symbolics.jacobian(ϕ, q))
    N_pusher = N[4:5]
    n_dir = N_pusher ./ sqrt(N_pusher[1]^2.0 + N_pusher[2]^2.0 + 1.0e-12)
    t_dir = [-n_dir[2]; n_dir[1]]

    r = p_pusher - p_block[1:2]
    m = r[1] * t_dir[2] - r[2] * t_dir[1]
    P = [t_dir[1]; t_dir[2]; m; -t_dir[1]; -t_dir[2]]

    return [P_block; transpose(P)]
end

function residual(model::PlanarPushV2, z, θ, κ)
    nq = model.nq
    nu = model.nu
    nc_impact = 1

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc_impact)]
    s1 = z[nq + nc_impact .+ (1:nc_impact)]

    ψ1 = z[nq + 2 * nc_impact .+ (1:5)]
    b1 = z[nq + 2 * nc_impact + 5 .+ (1:9)]
    sψ1 = z[nq + 2 * nc_impact + 5 + 9 .+ (1:5)]
    sb1 = z[nq + 2 * nc_impact + 5 + 9 + 5 .+ (1:9)]

    ϕ = ϕ_func(model, q2)
    N = vec(Symbolics.jacobian(ϕ, q2))
    P = P_func(model, q2)
    vT = P * (q2 - q1) ./ h[1]

    qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

    D1L1, D2L1 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm1, vm1)
    D1L2, D2L2 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm2, vm2)

    d = (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2
         + B_func(model, qm2) * u1
         + N * γ1[1]
         + transpose(P) * b1)

    [
     d;
     s1 .- ϕ;

     ψ1[1] .- model.mu_body * model.mass_block * model.gravity * h[1] * 0.25;
     ψ1[2] .- model.mu_body * model.mass_block * model.gravity * h[1] * 0.25;
     ψ1[3] .- model.mu_body * model.mass_block * model.gravity * h[1] * 0.25;
     ψ1[4] .- model.mu_body * model.mass_block * model.gravity * h[1] * 0.25;
     ψ1[5] .- model.mu_pusher * γ1[1];

     vT - sb1;

     γ1 .* s1 .- κ[1];
     cone_product([ψ1[1]; b1[1:2]], [sψ1[1]; sb1[1:2]]) - [κ[1]; 0.0; 0.0];
     cone_product([ψ1[2]; b1[2 .+ (1:2)]], [sψ1[2]; sb1[2 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
     cone_product([ψ1[3]; b1[4 .+ (1:2)]], [sψ1[3]; sb1[4 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
     cone_product([ψ1[4]; b1[6 .+ (1:2)]], [sψ1[4]; sb1[6 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
     cone_product([ψ1[5]; b1[8 .+ (1:1)]], [sψ1[5]; sb1[8 .+ (1:1)]]) - [κ[1]; 0.0];
    ]
end

function RoboDojo.indices_z(model::PlanarPushV2)
    nq = model.nq
    q = collect(1:nq)
    γ = collect(nq .+ (1:1))
    sγ = collect(nq + 1 .+ (1:1))
    ψ = collect(nq + 2 .+ (1:5))
    b = collect(nq + 2 + 5 .+ (1:9))
    sψ = collect(nq + 2 + 5 + 9 .+ (1:5))
    sb = collect(nq + 2 + 5 + 9 + 5 .+ (1:9))
    IndicesZ(q, γ, sγ, ψ, b, sψ, sb)
end

RoboDojo.nominal_configuration(model::PlanarPushV2) = [
    0.0;
    0.0;
    0.0;
    -(model.r_box + model.r_pusher) - 1.0e-8;
    0.0;
]

function RoboDojo.indices_optimization(model::PlanarPushV2)
    nq = model.nq
    nz = RoboDojo.num_var(model)
    IndicesOptimization(
        nz,
        nz,
        [collect(nq .+ (1:1)), collect(nq + 1 .+ (1:1))],
        [collect(nq .+ (1:1)), collect(nq + 1 .+ (1:1))],
        [
         [collect([collect(nq + 2 .+ (1:1)); collect(nq + 2 + 5 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 .+ (1:2))])],
         [collect([collect(nq + 2 + 1 .+ (1:1)); collect(nq + 2 + 5 + 2 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 1 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 2 .+ (1:2))])],
         [collect([collect(nq + 2 + 2 .+ (1:1)); collect(nq + 2 + 5 + 4 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 2 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 4 .+ (1:2))])],
         [collect([collect(nq + 2 + 3 .+ (1:1)); collect(nq + 2 + 5 + 6 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 3 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 6 .+ (1:2))])],
         [collect([collect(nq + 2 + 4 .+ (1:1)); collect(nq + 2 + 5 + 8 .+ (1:1))]), collect([collect(nq + 2 + 5 + 9 + 4 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 8 .+ (1:1))])],
        ],
        [
         [collect([collect(nq + 2 .+ (1:1)); collect(nq + 2 + 5 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 .+ (1:2))])],
         [collect([collect(nq + 2 + 1 .+ (1:1)); collect(nq + 2 + 5 + 2 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 1 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 2 .+ (1:2))])],
         [collect([collect(nq + 2 + 2 .+ (1:1)); collect(nq + 2 + 5 + 4 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 2 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 4 .+ (1:2))])],
         [collect([collect(nq + 2 + 3 .+ (1:1)); collect(nq + 2 + 5 + 6 .+ (1:2))]), collect([collect(nq + 2 + 5 + 9 + 3 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 6 .+ (1:2))])],
         [collect([collect(nq + 2 + 4 .+ (1:1)); collect(nq + 2 + 5 + 8 .+ (1:1))]), collect([collect(nq + 2 + 5 + 9 + 4 .+ (1:1)); collect(nq + 2 + 5 + 9 + 5 + 8 .+ (1:1))])],
        ],
        collect(1:(nq + 15)),
        collect(nq + 15 .+ (1:1)),
        collect(nq + 15 + 1 .+ (1:14)),
        [
         collect(nq + 15 + 1 .+ (1:3)),
         collect(nq + 15 + 4 .+ (1:3)),
         collect(nq + 15 + 7 .+ (1:3)),
         collect(nq + 15 + 10 .+ (1:3)),
         collect(nq + 15 + 13 .+ (1:2)),
        ],
        collect(nq + 15 .+ (1:15)),
    )
end

function RoboDojo.initialize_z!(z, model::PlanarPushV2, idx::IndicesZ, q)
    z[idx.q] .= q
    z[idx.γ] .= 1.0
    z[idx.sγ] .= 1.0
    z[idx.ψ] .= 1.0
    z[idx.b] .= 0.1
    z[idx.sψ] .= 1.0
    z[idx.sb] .= 0.1
end

RoboDojo.num_var(model::PlanarPushV2) = model.nq + 2 * 1 + 2 * 14
RoboDojo.friction_coefficients(model::PlanarPushV2{T}) where {T} = T[]

const r_box_v2 = 0.1
const r_pusher_v2 = 0.025

const cc1_v2 = [r_box_v2, r_box_v2]
const cc2_v2 = [-r_box_v2, r_box_v2]
const cc3_v2 = [r_box_v2, -r_box_v2]
const cc4_v2 = [-r_box_v2, -r_box_v2]
const contact_corner_offset_v2 = [cc1_v2, cc2_v2, cc3_v2, cc4_v2]

const gravity_v2 = 9.81
const mass_block_v2 = 1.0
const mass_pusher_v2 = 1.0
const mu_body_v2 = 0.5
const mu_pusher_v2 = 0.5
const inertia_v2 = 1.0 / 12.0 * mass_block_v2 * ((2.0 * r_box_v2)^2 + (2.0 * r_box_v2)^2)

planarpush_v2 = PlanarPushV2(
    5,      # nq
    2,      # nu
    0,      # nw
    5,      # nc
    1,      # nc_impact
    mass_block_v2,
    mass_pusher_v2,
    inertia_v2,
    mu_body_v2,
    mu_pusher_v2,
    gravity_v2,
    r_box_v2,
    r_pusher_v2,
    contact_corner_offset_v2,
)
