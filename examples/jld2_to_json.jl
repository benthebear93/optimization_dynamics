module OptimizationDynamics
using StaticArrays
    struct ContactTraj{T,nq,nu,nw,nc,nb,nz,nθ}
        H::Int
        h::T
        κ::Vector{T}
        q::Vector{Vector{T}}
        u::Vector{Vector{T}}
        w::Vector{Vector{T}}
        γ::Vector{Vector{T}}
        b::Vector{Vector{T}}
        z::Vector{Vector{T}}
        θ::Vector{Vector{T}}
        iq0::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
        iq1::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
        iu1::SizedArray{Tuple{nu},Int,1,1,Vector{Int}}
        iw1::SizedArray{Tuple{nw},Int,1,1,Vector{Int}}
        iq2::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
        iγ1::SizedArray{Tuple{nc},Int,1,1,Vector{Int}}
        ib1::SizedArray{Tuple{nb},Int,1,1,Vector{Int}}
    end
end

using JLD2
using JSON3
using StaticArrays

function tojsonable(traj)
    return Dict(
        "H"      => traj.H,
        "h"      => traj.h,
        "kappa"  => traj.κ,            # κ
        "q"      => traj.q,
        "u"      => traj.u,
        "w"      => traj.w,
        "gamma"  => traj.γ,            # γ
        "b"      => traj.b,
        "z"      => traj.z,
        "theta"  => traj.θ,            # θ
        "iq0"    => collect(traj.iq0),
        "iq1"    => collect(traj.iq1),
        "iu1"    => collect(traj.iu1),
        "iw1"    => collect(traj.iw1),
        "iq2"    => collect(traj.iq2),
        "igamma1"=> collect(traj.iγ1),
        "ib1"    => collect(traj.ib1),

        "nq"     => length(traj.iq0),
        "nu"     => length(traj.iu1),
        "nw"     => length(traj.iw1),
        "nc"     => length(traj.iγ1),
        "nb"     => length(traj.ib1),
        "nz"     => length(first(traj.z)),
        "nθ"     => length(first(traj.θ)),
    )
end

data = jldopen("/home/haegu/optimization_dynamics/examples/pusher_ref_traj.jld2", "r") do f
    read(f, "traj")
end

jsonable = tojsonable(data)
open("pusher_ref_traj.json", "w") do io
    JSON3.write(io, jsonable; indent=2)
end

