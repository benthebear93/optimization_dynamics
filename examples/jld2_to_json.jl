# module OptimizationDynamics
# using StaticArrays
#     struct ContactTraj{T,nq,nu,nw,nc,nb,nz,nθ}
#         H::Int
#         h::T
#         κ::Vector{T}
#         q::Vector{Vector{T}}
#         u::Vector{Vector{T}}
#         w::Vector{Vector{T}}
#         γ::Vector{Vector{T}}
#         b::Vector{Vector{T}}
#         z::Vector{Vector{T}}
#         θ::Vector{Vector{T}}
#         iq0::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
#         iq1::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
#         iu1::SizedArray{Tuple{nu},Int,1,1,Vector{Int}}
#         iw1::SizedArray{Tuple{nw},Int,1,1,Vector{Int}}
#         iq2::SizedArray{Tuple{nq},Int,1,1,Vector{Int}}
#         iγ1::SizedArray{Tuple{nc},Int,1,1,Vector{Int}}
#         ib1::SizedArray{Tuple{nb},Int,1,1,Vector{Int}}
#     end
# end

# using JLD2
# using JSON3
# using StaticArrays

# function tojsonable(traj)
#     return Dict(
#         "H"      => traj.H,
#         "h"      => traj.h,
#         "kappa"  => traj.κ,            # κ
#         "q"      => traj.q,
#         "u"      => traj.u,
#         "w"      => traj.w,
#         "gamma"  => traj.γ,            # γ
#         "b"      => traj.b,
#         "z"      => traj.z,
#         "theta"  => traj.θ,            # θ
#         "iq0"    => collect(traj.iq0),
#         "iq1"    => collect(traj.iq1),
#         "iu1"    => collect(traj.iu1),
#         "iw1"    => collect(traj.iw1),
#         "iq2"    => collect(traj.iq2),
#         "igamma1"=> collect(traj.iγ1),
#         "ib1"    => collect(traj.ib1),

#         "nq"     => length(traj.iq0),
#         "nu"     => length(traj.iu1),
#         "nw"     => length(traj.iw1),
#         "nc"     => length(traj.iγ1),
#         "nb"     => length(traj.ib1),
#         "nz"     => length(first(traj.z)),
#         "nθ"     => length(first(traj.θ)),
#     )
# end

# data = jldopen("/home/haegu/optimization_dynamics/examples/pusher_ref_traj.jld2", "r") do f
#     read(f, "traj")
# end

# jsonable = tojsonable(data)
# open("pusher_ref_traj.json", "w") do io
#     JSON3.write(io, jsonable; indent=2)
# end

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
using OrderedCollections: OrderedDict

function tojsonable(traj)
    od = OrderedDict{String,Any}()
    # --- 원하는 출력 순서대로 삽입 ---
    od["H"]        = traj.H
    od["h"]        = traj.h
    od["kappa"]    = traj.κ            # κ
    od["q"]        = traj.q
    od["u"]        = traj.u
    od["w"]        = traj.w
    od["gamma"]    = traj.γ            # γ
    od["b"]        = traj.b
    od["z"]        = traj.z
    od["theta"]    = traj.θ            # θ

    od["iq0"]      = collect(traj.iq0)
    od["iq1"]      = collect(traj.iq1)
    od["iu1"]      = collect(traj.iu1)
    od["iw1"]      = collect(traj.iw1)
    od["iq2"]      = collect(traj.iq2)
    od["igamma1"]  = collect(traj.iγ1)
    od["ib1"]      = collect(traj.ib1)

    od["nq"]       = length(traj.iq0)
    od["nu"]       = length(traj.iu1)
    od["nw"]       = length(traj.iw1)
    od["nc"]       = length(traj.iγ1)
    od["nb"]       = length(traj.ib1)
    od["nz"]       = length(first(traj.z))
    od["nθ"]       = length(first(traj.θ))
    return od
end

data = jldopen("/home/haegu/optimization_dynamics/examples/pusher_ref_traj.jld2", "r") do f
    read(f, "traj")
end

jsonable = tojsonable(data)
open("pusher_ref_traj.json", "w") do io
    JSON3.write(io, jsonable; indent=2)
end
