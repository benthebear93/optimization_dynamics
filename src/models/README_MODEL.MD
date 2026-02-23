# Models Overview

The `src/models` directory contains contact/friction-based pushing dynamics models, plus simulation and visualization code.

## Folder Descriptions

- `planar_push`: Base planar pushing model with a free block (x, y, yaw) and a single point pusher.
- `fixed_planar_push`: Simplified planar pushing model where block translation is fixed and only rotation is free.
- `line_planar_push`: Model with a two-point line pusher (connected by a spring-damper) pushing a fixed block.
- `line_planar_push_xy`: Extension of `line_planar_push` that also allows block translation in x/y.
- `tipover_push`: 2D x-z tip-over model with dynamic pusher contact.

## Common File Layout (Most Models)

- `model.jl`: Model parameters, mass/force/contact functions, and residual definition.
- `simulator.jl`: Simulator setup and runtime helpers.
- `visuals.jl`: MeshCat-based visualization utilities.
- `codegen.jl`: Symbolics-based residual/Jacobian code generation and JLD2 export.
