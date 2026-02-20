append!(empty!(LOAD_PATH), Base.DEFAULT_LOAD_PATH)
using Pkg

################################################################################
# Generate notebooks
################################################################################
exampledir = joinpath(@__DIR__, "..", "examples")
Pkg.activate(exampledir)
Pkg.instantiate()
include(joinpath(exampledir, "generate_notebooks.jl"))

################################################################################
# Build simulation environments
################################################################################
pkgdir = joinpath(@__DIR__, "..")
Pkg.activate(pkgdir)

using JLD2 
using Symbolics
using LinearAlgebra
using Scratch 
using RoboDojo 
using Rotations
import RoboDojo: Model, lagrangian_derivatives, IndicesZ, cone_product

# hopper from RoboDojo.jl 

# planar push 
include("../src/models/planar_push/model.jl")
include("../src/models/planar_push/simulator.jl")
include("../src/models/planar_push/codegen.jl")

# fixed planar push 
include("../src/models/fixed_planar_push/model.jl")
include("../src/models/fixed_planar_push/simulator.jl")
include("../src/models/fixed_planar_push/codegen.jl")

# line planar push 
include("../src/models/line_planar_push/model.jl")
include("../src/models/line_planar_push/simulator.jl")
include("../src/models/line_planar_push/codegen.jl")
