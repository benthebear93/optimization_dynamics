function _create_planar_push!(vis, model::LinePlanarPush;
        i = 1,
        r = 0.1,#0.325,
        r_pusher = 0.025,#0.08125,
        tl = 1.0,
        box_color = Colors.RGBA(0.569, 0.573, 0.573, 1),
        pusher_color = Colors.RGBA(0.569, 0.573, 0.573, 1))

    r_box = r - r_pusher

    setobject!(vis["box_$i"], GeometryBasics.Rect(Vec(-1.0 * r_box,
		-1.0 * r_box,
		-1.0 * r_box),
		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
		MeshPhongMaterial(color = box_color))

    setobject!(vis["pusher_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = pusher_color))
    setobject!(vis["pusher_2_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = pusher_color))
end

function _create_tail!(vis, model::LinePlanarPush;
        i = 1,
        r = 0.1, #0.325,
        r_pusher = 0.025, #0.08125,
        tl = 0.5)

    r_box = r - r_pusher
    box_tail_color = Colors.RGBA(0.678, 0.675, 0.675, 0.1)
    setobject!(vis["box_tail_$i"], GeometryBasics.Rect(Vec(-1.0 * r_box,
		-1.0 * r_box,
		-1.0 * r_box),
		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
		MeshPhongMaterial(color = box_tail_color))
    tail_color = Colors.RGBA(0.502, 1.0, 0.490, 0.1)
    setobject!(vis["pusher_tail_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = tail_color))
    setobject!(vis["pusher_tail_2_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = tail_color))
end

function _set_planar_push!(vis, model::LinePlanarPush, q;
    i = 1)
    settransform!(vis["box_$i"],
		compose(Translation(0, 0, 0.01 * i), LinearMap(RotZ(q[1]))))
    settransform!(vis["pusher_$i"], Translation(q[2], q[3], 0.01 * i))
    settransform!(vis["pusher_2_$i"], Translation(q[4], q[5], 0.01 * i))
end

function _set_planar_box_tail!(vis, model::LinePlanarPush, q;
    i = 1)
    settransform!(vis["box_tail_$i"],
        compose(Translation(0, 0, 0.01 * i), LinearMap(RotZ(q[1]))))
end

function _set_planar_pusher_tail!(vis, model::LinePlanarPush, q;
    i = 1)
    settransform!(vis["pusher_tail_$i"], Translation(q[2], q[3], 0.01 * i))
    settransform!(vis["pusher_tail_2_$i"], Translation(q[4], q[5], 0.01 * i))
end

function visualize!(vis, model::LinePlanarPush, q;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = Colors.RGBA(0.569, 0.573, 0.573, 1),
        pusher_color = Colors.RGBA(0.569, 0.573, 0.573, 1),
        Δt = 0.1)

	default_background!(vis)

    _create_planar_push!(vis, model,
        i = i,
        r = r,
        r_pusher = r_pusher,
        tl = tl,
        box_color = box_color,
        pusher_color = pusher_color)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

	T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
            _set_planar_push!(vis, model, q[t])
        end
    end

	settransform!(vis["/Cameras/default"],
    compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)


    MeshCat.setanimation!(vis, anim)
end


function visualize_with_trail!(vis, model::LinePlanarPush, q;
        i = 1,
        r = 0.1, #0.325,
        r_pusher = 0.025, #0.08125,
        tl = 1.0,
        box_color = Colors.RGBA(0.569, 0.573, 0.573, 1),
        pusher_color = Colors.RGBA(0.569, 0.573, 0.573, 1),
        Δt = 0.1,
        trail_length = 1) 

    default_background!(vis)

    # create box (static)
    _create_planar_push!(vis, model,
        i = i,
        r = r,
        r_pusher = r_pusher,
        tl = tl,
        box_color = box_color,
        pusher_color = pusher_color)

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(q)
    for t = 1:T-1
        _create_tail!(vis, model, i=t, r_pusher=r_pusher, tl=0.5)
        MeshCat.atframe(anim, t) do
            _set_planar_push!(vis, model, q[t]; i=1)
            _set_planar_box_tail!(vis, model, q[t]; i=t)
            _set_planar_pusher_tail!(vis, model, q[t]; i=t)
        end
    end

    settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, 0.0, 20.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)

    MeshCat.setanimation!(vis, anim)
end
