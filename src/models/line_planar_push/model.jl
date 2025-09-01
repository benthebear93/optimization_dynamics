"""
    Fixed planar push block
        particle with contacts at each corner
"""
struct LinePlanarPush{T} <: Model{T}
	# dimensions
	nq::Int # generalized coordinates
	nu::Int # controls
	nw::Int # parameters
	nc::Int # contact points

    mass_block::T
	mass_pusher::T

    inertia::T
    μ_surface::Vector{T}
	μ_pusher::T
    gravity::T

    contact_corner_offset::Vector{Vector{T}}
end

# Kinematics
r_dim = 0.1

function sd_2d_box(p, pose)
	x, y, θ = pose
	Δ = rotation_matrix(-θ) * (p - pose[1:2])
	s = 10
	sum(Δ.^s)^(1/s) - r_dim
end

# contact corner
cc1 = [r_dim, r_dim]
cc2 = [-r_dim, r_dim]
cc3 = [r_dim, -r_dim]
cc4 = [-r_dim, -r_dim]

contact_corner_offset = [cc1, cc2, cc3, cc4]

# Parameters
μ_surface = 0.5  # coefficient of friction
μ_pusher = 0.5
gravity = 9.81
mass_block = 1.0   # mass
mass_pusher = 10.0
inertia = 1.0 / 12.0 * mass_block * ((2.0 * r_dim)^2 + (2.0 * r_dim)^2)
L = 0.05
k_spring = 300
c_damping = 15
# rnd = 0.01
# dim = [r_dim, r_dim]
# dim_rnd = [r_dim - rnd, r_dim - rnd]

# Methods
M_func(model::LinePlanarPush, q) = Diagonal([model.inertia, model.mass_pusher, model.mass_pusher, model.mass_pusher, model.mass_pusher])

function C_func(model::LinePlanarPush, q, q̇)
	[0.0, 0.0, 0.0, 0.0, 0.0]
end

function rotation_matrix(x)
	[cos(x) -sin(x); sin(x) cos(x)]
end

function ϕ_func(model::LinePlanarPush, q)
    p_block = [0, 0, q[1]]
	p_pusher1 = [q[2], q[3]]
	p_pusher2 = [q[4], q[5]]

	sdf1 = sd_2d_box(p_pusher1, p_block)
	sdf2 = sd_2d_box(p_pusher2, p_block)
    return [[sdf1], [sdf2]]
end

function B_func(model::LinePlanarPush, q)
	[0.0 0.0 0.0 0.0;
	 1.0 0.0 0.0 0.0;
	 0.0 1.0 0.0 0.0;
	 0.0 0.0 1.0 0.0;
	 0.0 0.0 0.0 1.0]
end

function N_func(model::LinePlanarPush, q)
	ϕ = ϕ_func(model, q) 
	vec(Symbolics.jacobian(ϕ, q))
end

function p_func(model, x)
    pos = [0, 0]
    θ = x[1]
    R = rotation_matrix(θ)

    [(pos + R * model.contact_corner_offset[1])[1:2];
     (pos + R * model.contact_corner_offset[2])[1:2];
     (pos + R * model.contact_corner_offset[3])[1:2];
     (pos + R * model.contact_corner_offset[4])[1:2]]
end

function P_func(model::LinePlanarPush, q)
	pf = p_func(model, q)
	P_block = Symbolics.jacobian(pf, q)

	p_block = [0, 0, q[1]]
	p_pusher1 = q[2:3] 
	p_pusher2 = q[4:5] 

	ϕ  = ϕ_func(model, q)
	N1 = vec(Symbolics.jacobian(ϕ[1], q))
	N2 = vec(Symbolics.jacobian(ϕ[2], q))

	N_pusher1 = N1[2:3]
	N_pusher2 = N2[2:3]

	n_dir1 = N_pusher1 ./ sqrt(N_pusher1[1]^2.0 + N_pusher1[2]^2.0)
	t_dir1 = [-n_dir1[2]; n_dir1[1]]

	n_dir2 = N_pusher2 ./ sqrt(N_pusher2[1]^2.0 + N_pusher2[2]^2.0)
	t_dir2 = [-n_dir2[2]; n_dir2[1]]

	r1 = p_pusher1 - p_block[1:2]
	m1 = r1[1] * t_dir1[2] - r1[2] * t_dir1[1]

	r2 = p_pusher2 - p_block[1:2]
	m2 = r2[1] * t_dir2[2] - r2[2] * t_dir2[1]

	P1 = [m1; -t_dir1[1]; -t_dir1[2]; 0; 0]
	P2 = [m2; 0; 0; -t_dir2[1]; -t_dir2[2]]

	return [P_block; transpose(P1); transpose(P2)]
end

function spring_damper_force(q, q̇)
	px1, py1 = q[2:3]
	px2, py2 = q[4:5]
	vx1, vy1 = q̇[2:3]
	vx2, vy2 = q̇[4:5]

	delta_x = px2 - px1
	delta_y = py2 - py1
	dist = sqrt(delta_x^2.0 + delta_y^2.0)
	dist_error = dist - L
	dist_dot = (delta_x * (vx2 - vx1) + delta_y * (vy2 - vy1)) / dist
	
	force_mag = -k_spring * dist_error - c_damping * dist_dot
	
	unit_vec = [delta_x / dist, delta_y / dist]
	
	f1 = force_mag * unit_vec
	f2 = -force_mag * unit_vec
	
	return [0, f1[1], f1[2], f2[1], f2[2]]
end

function residual(model, z, θ, κ)
    nq = model.nq
    nu = model.nu
    nc_impact = 2

    q0 = θ[1:nq]
    q1 = θ[nq .+ (1:nq)]
    u1 = θ[2nq .+ (1:nu)]
    h = θ[2nq + nu .+ (1:1)]

    q2 = z[1:nq]
    γ1 = z[nq .+ (1:nc_impact)]
	s1 = z[nq + nc_impact .+ (1:nc_impact)]

	ψ1 = z[nq + 2 * nc_impact .+ (1:6)]
	b1 = z[nq + 2 * nc_impact + 6 .+ (1:10)]
	sψ1 = z[nq + 2 * nc_impact + 6 + 10 .+ (1:6)]
	sb1 = z[nq + 2 * nc_impact + 6 + 10 + 6 .+ (1:10)]

	ϕ = ϕ_func(model, q2)
	N1 = vec(Symbolics.jacobian(ϕ[1], q2))
	N2 = vec(Symbolics.jacobian(ϕ[2], q2))
	@show size(N1)
	@show size(N2)
	@show size(γ1)
	N = hcat(N1, N2)  # Creates a 5x2 matrix
	@show size(N)
	@show size(γ1)

	# λ1 = [b1; γ1]
	P = P_func(model, q2)
    vT = P * (q2 - q1) ./ h[1]

	qm1 = 0.5 * (q0 + q1)
    vm1 = (q1 - q0) / h[1]
    qm2 = 0.5 * (q1 + q2)
    vm2 = (q2 - q1) / h[1]

	D1L1, D2L1 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm1, vm1)
	D1L2, D2L2 = lagrangian_derivatives(a -> M_func(model, a), (a, b) -> C_func(model, a, b), qm2, vm2)
	f_spring_damper = spring_damper_force(qm2, vm2)
    d = (0.5 * h[1] * D1L1 + D2L1 + 0.5 * h[1] * D1L2 - D2L2#
            + B_func(model, qm2) * u1
            + N * γ1
            + transpose(P) * b1)
			+ f_spring_damper

    [
	 d;
    
	 s1[1] .- ϕ[1];

	 s1[2] .- ϕ[2];

	 ψ1[1] .- model.μ_surface[1] * model.mass_block * model.gravity * h[1] * 0.25;

	 ψ1[2] .- model.μ_surface[2] * model.mass_block * model.gravity * h[1] * 0.25;

	 ψ1[3] .- model.μ_surface[3] * model.mass_block * model.gravity * h[1] * 0.25;

	 ψ1[4] .- model.μ_surface[4] * model.mass_block * model.gravity * h[1] * 0.25;

	 ψ1[5] .- model.μ_pusher * γ1[1];

	 ψ1[6] .- model.μ_pusher * γ1[2];

	 vT - sb1;

	 γ1[1] .* s1[1] .- κ[1];
	 γ1[2] .* s1[2] .- κ[1];
	 cone_product([ψ1[1]; b1[1:2]], [sψ1[1]; sb1[1:2]]) - [κ[1]; 0.0; 0.0];
	 cone_product([ψ1[2]; b1[2 .+ (1:2)]], [sψ1[2]; sb1[2 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
	 cone_product([ψ1[3]; b1[4 .+ (1:2)]], [sψ1[3]; sb1[4 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
	 cone_product([ψ1[4]; b1[6 .+ (1:2)]], [sψ1[4]; sb1[6 .+ (1:2)]]) - [κ[1]; 0.0; 0.0];
	 cone_product([ψ1[5]; b1[8 .+ (1:1)]], [sψ1[5]; sb1[8 .+ (1:1)]]) - [κ[1]; 0.0];
	 cone_product([ψ1[6]; b1[9 .+ (1:1)]], [sψ1[6]; sb1[9 .+ (1:1)]]) - [κ[1]; 0.0];
    ]
end

# Dimensions
nq = 5 # configuration dimension
nu = 4 # control dimension
nc = 5 # number of contact points
nc_impact = 1
nf = 3 # number of faces for friction cone pyramid
nb = 10 #(nc - nc_impact) * nf + (nf - 1) * nc_impact

lineplanarpush = LinePlanarPush(nq, nu, 0, nc,
			mass_block, mass_pusher, 
			inertia, [μ_surface for i = 1:nc], μ_pusher, gravity,
			contact_corner_offset)
