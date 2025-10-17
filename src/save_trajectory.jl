using JLD2
using StaticArrays

struct ContactTraj{T,nq,nu,nw,nc,nb,nz,nθ}
    H::Int
    h::T
    κ::Vector{T}
    q::Vector{Vector{T}}   # trajectory of q's   length=H+2
    u::Vector{Vector{T}}   # trajectory of u's   length=H
    w::Vector{Vector{T}}   # trajectory of w's   length=H
    γ::Vector{Vector{T}}   # trajectory of γ's   length=H
    b::Vector{Vector{T}}   # trajectory of b's   length=H
    z::Vector{Vector{T}}   # trajectory of z's   length=H
    θ::Vector{Vector{T}}   # trajectory of θ's   length=H
    iq0::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
    iq1::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
    iu1::SizedArray{Tuple{nu},Int,1,1,Vector{Int}}
    iw1::SizedArray{Tuple{nw},Int,1,1,Vector{Int}}
    iq2::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
    iγ1::SizedArray{Tuple{nc},Int,1,1,Vector{Int}}
    ib1::SizedArray{Tuple{nb},Int,1,1,Vector{Int}}
end

function save_trajectory(gait_path::String, x_sol, u_sol, gamma_hist, b_hist, ip_z_hist, ip_θ_hist, w, model, h::Float64, T::Int)

    nq = model[1].nx ÷ 2  #model[1].nx ÷ 2  # planarpush.nq
    nu = model[1].nu       # planarpush.nu
    nw = model[1].nw       # num_w (0 in this case)
    nc = model[1].nc       # nc_impact (1 in this case)
    nb = model[1].nc - model[1].nc_impact          
    nz = length(ip_z_hist[1])  
    nθ = length(ip_θ_hist[1])

    H = T - 1
    println("H ", H, "len xsol ", length(x_sol), " ", model[1].nx)
    @assert length(x_sol) >= H + 1 "x_sol must have at least H+2 elements"
    @assert length(u_sol) == H "u_sol must have H elements"
    @assert length(w) >= H "w must have at least H elements"
    @assert length(gamma_hist) == H "gamma_hist must have H elements"
    @assert length(b_hist) == H "b_hist must have H elements"
    @assert length(ip_z_hist) == H "ip_z_hist must have H elements"
    @assert length(ip_θ_hist) == H "ip_θ_hist must have H elements"

    # ContactTraj
    κ = fill(2e-8, H)
    q = [x_sol[t][1:nq] for t in 1:(H)] 
    u = deepcopy(u_sol)
    w_traj = deepcopy(w[1:H])
    γ = [Vector{Float64}(g) for g in gamma_hist] 
    b = [Vector{Float64}(b) for b in b_hist]  
    z = [Vector{Float64}(z) for z in ip_z_hist]  
    θ = [Vector{Float64}(θ) for θ in ip_θ_hist] 

    iq0 = SizedArray{Tuple{nq},Int,1,1}(collect(1:nq))
    iq1 = SizedArray{Tuple{nq},Int,1,1}(collect(nq+1:2*nq))
    iu1 = SizedArray{Tuple{nu},Int,1,1}(collect(2*nq+1:2*nq + nu))
    iw1 = SizedArray{Tuple{nw},Int,1,1}(collect(2*nq + nu +1: 2*nq + nu + nw))
    iq2 = SizedArray{Tuple{nq},Int,1,1}(collect(1:nq))
    iγ1 = SizedArray{Tuple{nc},Int,1,1}(collect(nq + 1: nq + nc))
    ib1 = SizedArray{Tuple{nb},Int,1,1}(collect(nq + nc +1:nq + nc + nb))

    # @show iq0
    # @show iq1
    # @show iu1
    # @show iw1
    # @show iq2
    # @show iγ1
    # @show ib1
    
    traj = ContactTraj{Float64,nq,nu,nw,nc,nb,nz,nθ}(
        H, h, κ, q, u, w_traj, γ, b, z, θ, iq0, iq1, iu1, iw1, iq2, iγ1, ib1
    )
    
    jldsave(gait_path; traj = traj)      


    return nothing
end