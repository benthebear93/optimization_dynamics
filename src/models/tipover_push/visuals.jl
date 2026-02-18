function _default_background_tipover!(vis)
    OptimizationDynamics.MeshCat.setvisible!(vis["/Background"], true)
    OptimizationDynamics.MeshCat.setprop!(vis["/Background"], "top_color", OptimizationDynamics.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    OptimizationDynamics.MeshCat.setprop!(vis["/Background"], "bottom_color", OptimizationDynamics.Colors.RGBA(1.0, 1.0, 1.0, 1.0))
    OptimizationDynamics.MeshCat.setvisible!(vis["/Axes"], false)
end

function _create_tipover_push!(vis, model::TipOverPush;
        i = 1,
        box_color = OptimizationDynamics.Colors.RGBA(0.494, 0.863, 0.604, 1.0),
        pusher_color = OptimizationDynamics.Colors.RGBA(0.91, 0.3, 0.24, 0.95))

    OptimizationDynamics.MeshCat.setobject!(vis["box_$i"],
        OptimizationDynamics.GeometryBasics.Rect(
            OptimizationDynamics.GeometryBasics.Vec(-model.box_half_width, -0.04, -model.box_half_height),
            OptimizationDynamics.GeometryBasics.Vec(2.0 * model.box_half_width, 0.08, 2.0 * model.box_half_height),
        ),
        OptimizationDynamics.MeshCat.MeshPhongMaterial(color = box_color))

    OptimizationDynamics.MeshCat.setobject!(vis["pusher_$i"],
        OptimizationDynamics.GeometryBasics.HyperSphere(OptimizationDynamics.GeometryBasics.Point3f0(0.0), convert(Float32, model.pusher_radius)),
        OptimizationDynamics.MeshCat.MeshPhongMaterial(color = pusher_color))
end

function _set_tipover_push!(vis, model::TipOverPush, q; i = 1)
    OptimizationDynamics.MeshCat.settransform!(vis["box_$i"],
        OptimizationDynamics.CoordinateTransformations.compose(
            OptimizationDynamics.CoordinateTransformations.Translation(q[1], 0.0, q[3]),
            OptimizationDynamics.CoordinateTransformations.LinearMap(OptimizationDynamics.Rotations.RotY(q[4]))))
    OptimizationDynamics.MeshCat.settransform!(vis["pusher_$i"], OptimizationDynamics.CoordinateTransformations.Translation(q[5], 0.0, q[7]))
end

function visualize!(vis, model::TipOverPush, q;
        i = 1,
        fix_camera = false,
        cam_zoom = 35.0,
        Δt = 0.05)

    _default_background_tipover!(vis)
    OptimizationDynamics.MeshCat.setvisible!(vis["/Grid"], true)

    _create_tipover_push!(vis, model, i = i)

    anim = OptimizationDynamics.MeshCat.Animation(convert(Int, floor(1.0 / Δt)))
    for t = 1:length(q)
        OptimizationDynamics.MeshCat.atframe(anim, t) do
            _set_tipover_push!(vis, model, q[t], i = i)
        end
    end

    if fix_camera
        # Optional fixed view aligned so x-z motion appears 2D.
        OptimizationDynamics.MeshCat.settransform!(vis["/Cameras/default"],
            OptimizationDynamics.CoordinateTransformations.compose(
                OptimizationDynamics.CoordinateTransformations.Translation(0.0, -50.0, -1.0),
                OptimizationDynamics.CoordinateTransformations.LinearMap(OptimizationDynamics.Rotations.RotZ(-pi / 2.0))))
        OptimizationDynamics.MeshCat.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", cam_zoom)
    end

    OptimizationDynamics.MeshCat.setanimation!(vis, anim)
end
