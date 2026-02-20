module OptimizationDynamics

using CSV
using DataFrames
using LinearAlgebra 
using BenchmarkTools
using Symbolics 
using IfElse
using JLD2
import JLD2: load 
using DirectTrajectoryOptimization
using IterativeLQR
using MeshCat 
using Colors
using CoordinateTransformations 
using GeometryBasics
using Rotations 
using RoboDojo 
import RoboDojo: LinearSolver, LUSolver, Model, ResidualMethods, Space, Disturbances, IndicesZ, InteriorPoint, EmptySolver, Policy, Trajectory, GradientTrajectory, InteriorPointOptions, IndicesOptimization, interior_point, interior_point_solve!, bilinear_violation, residual_violation, general_correction_term!, r!, rz!, rθ!, linear_solve!, lu_solver, empty_policy, empty_disturbances, friction_coefficients, SimulatorStatistics, SimulatorOptions, indices_θ, num_data, initialize_z!, initialize_θ!, indices_z, indices_θ, simulate!, policy, process!, Simulator, cone_product, lagrangian_derivatives, Indicesθ
using Scratch 
using Plots
using StaticArrays

include("save_trajectory.jl")
export save_trajectory
export LinearSolver, LUSolver, Model, ResidualMethods, Space, Disturbances, IndicesZ, InteriorPoint, EmptySolver, Policy, Trajectory, GradientTrajectory, InteriorPointOptions, IndicesOptimization, interior_point, interior_point_solve!, bilinear_violation, residual_violation, general_correction_term!, r!, rz!, rθ!, linear_solve!, lu_solver, empty_policy, empty_disturbances, friction_coefficients, SimulatorStatistics, SimulatorOptions, indices_θ, num_data, initialize_z!, initialize_θ!, indices_z, indices_θ, simulate!, policy, process!, Simulator, cone_product, lagrangian_derivatives, Indicesθ

export 
    load

export 
    Visualizer, render, open, visualize!, visualize_with_trail!

include("dynamics.jl")
include("ls.jl")
include("gradient_bundle.jl")
export 
    ImplicitDynamics, f, fx, fu, state_to_configuration,
    f_gb, fx_gb, fu_gb, f_debug

export 
    ϕ_func

# # hopper from RoboDojo.jl 

# planar push 
include("../src/models/planar_push/model.jl")
include("../src/models/planar_push/simulator.jl")
include("../src/models/planar_push/visuals.jl")
path_planarpush = @get_scratch!("planarpush")
# @show r_pp_func
@load joinpath(path_planarpush, "residual.jld2") r_pp_func rz_pp_func rθ_pp_func rz_pp_array rθ_pp_array

# fixed planar push 
include("../src/models/fixed_planar_push/model.jl")
include("../src/models/fixed_planar_push/simulator.jl")
include("../src/models/fixed_planar_push/visuals.jl")
path_fixed_planarpush = @get_scratch!("fixedplanarpush")
@show path_fixed_planarpush
@load joinpath(path_fixed_planarpush, "residual.jld2") r_fpp_func rz_fpp_func rθ_fpp_func rz_fpp_array rθ_fpp_array

# line planar push 
include("../src/models/line_planar_push/model.jl")
include("../src/models/line_planar_push/simulator.jl")
include("../src/models/line_planar_push/visuals.jl")
path_line_planarpush = @get_scratch!("lineplanarpush")
@load joinpath(path_line_planarpush, "residual.jld2") r_lpp_func rz_lpp_func rθ_lpp_func rz_lpp_array rθ_lpp_array

# line planar push with translation
include("../src/models/line_planar_push_xy/model.jl")
include("../src/models/line_planar_push_xy/simulator.jl")
include("../src/models/line_planar_push_xy/visuals.jl")
path_line_planarpush_xy = @get_scratch!("lineplanarpush_xy")
@load joinpath(path_line_planarpush_xy, "residual.jld2") r_lppxy_func rz_lppxy_func rθ_lppxy_func rz_lppxy_array rθ_lppxy_array

include("../src/models/visualize.jl")

export 
    planarpush, 
    fixedplanarpush, lineplanarpush, lineplanarpush_xy

export 
    r_pp_func, rz_pp_func, rθ_pp_func, rz_pp_array, rθ_pp_array,
    r_fpp_func,rz_fpp_func, rθ_fpp_func, rz_fpp_array, rθ_fpp_array,
    r_lpp_func,rz_lpp_func, rθ_lpp_func, rz_lpp_array, rθ_lpp_array,
    r_lppxy_func,rz_lppxy_func, rθ_lppxy_func, rz_lppxy_array, rθ_lppxy_array
    # r_proj_func, rz_proj_func, rθ_proj_func, rz_proj_array, rθ_proj_array

end # module
