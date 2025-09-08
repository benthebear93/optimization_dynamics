function _create_planar_push!(vis, model::FixedPlanarPush;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = Colors.RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = Colors.RGBA(0.5, 0.5, 0.5, tl))

    r_box = r - r_pusher

    setobject!(vis["box_$i"], GeometryBasics.Rect(Vec(-1.0 * r_box,
		-1.0 * r_box,
		-1.0 * r_box),
		Vec(2.0 * r_box, 2.0 * r_box, 2.0 * r_box)),
		MeshPhongMaterial(color = box_color))

    setobject!(vis["pusher_$i"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r_box), r_pusher),
        MeshPhongMaterial(color = pusher_color))
end

function _set_planar_push!(vis, model::FixedPlanarPush, q;
    i = 1)
    settransform!(vis["box_$i"],
		compose(Translation(0, 0, 0.01 * i), LinearMap(RotZ(q[1]))))
    settransform!(vis["pusher_$i"], Translation(q[2], q[3], 0.01 * i))
end

function visualize!(vis, model::FixedPlanarPush, q;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = Colors.RGBA(0.0, 0.0, 0.0, tl),
        pusher_color = Colors.RGBA(0.5, 0.5, 0.5, tl),
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

using Colors

function visualize_with_fadeout!(vis, model::FixedPlanarPush, q;
        i = 1,
        r = 0.1,
        r_pusher = 0.025,
        tl = 1.0,
        box_color = RGBA{Float32}(0.0, 0.0, 0.0, tl),
        pusher_color = RGBA{Float32}(0.5, 0.5, 0.5, tl),
        Δt = 0.1,
        fade_steps = 10)  # 몇 프레임 동안 서서히 사라질지

    default_background!(vis)

    # box, pusher 생성
    _create_planar_push!(vis, model,
        i = i,
        r = r,
        r_pusher = r_pusher,
        tl = tl,
        box_color = box_color,
        pusher_color = pusher_color)

    # ghost trail object 하나 추가
    setobject!(vis["pusher_trail_$(i)"],
        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r - r_pusher), r_pusher),
        MeshPhongMaterial(color = RGBA{Float32}(0.5, 0.5, 0.5, 0.0)))  # 처음엔 투명

    anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

    T = length(q)
    for t = 1:T-1
        MeshCat.atframe(anim, t) do
            # 현재 푸셔 위치 업데이트
            _set_planar_push!(vis, model, q[t]; i=i)

            # ghost trail 위치를 이전 프레임 위치로 설정
            if t > 1
                settransform!(vis["pusher_trail_$(i)"], Translation(q[t-1][2], q[t-1][3], 0.01 * i))
            end

            # ghost 투명도 fade-out
            for k = 0:fade_steps-1
                α = max(0.0, 0.5 * (1 - k/fade_steps))  # 점점 줄어듦
                MeshCat.atframe(anim, t+k) do
                    setobject!(vis["pusher_trail_$(i)"],
                        Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r - r_pusher), r_pusher),
                        MeshPhongMaterial(color = RGBA{Float32}(0.5, 0.5, 0.5, α)))
                end
            end
        end
    end

    # 카메라 위치 고정
    settransform!(vis["/Cameras/default"],
        compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
    setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)

    MeshCat.setanimation!(vis, anim)
end


# function visualize_with_trail!(vis, model::FixedPlanarPush, q;
#         i = 1,
#         r = 0.1,
#         r_pusher = 0.025,
#         tl = 1.0,
#         box_color = Colors.RGBA(0.0, 0.0, 0.0, tl),
#         pusher_color = Colors.RGBA(0.5, 0.5, 0.5, tl),
#         Δt = 0.1,
#         trail_length = 5)  # 잔상 개수

#     default_background!(vis)

#     # create box (static)
#     _create_planar_push!(vis, model,
#         i = i,
#         r = r,
#         r_pusher = r_pusher,
#         tl = tl,
#         box_color = box_color,
#         pusher_color = pusher_color)

#     anim = MeshCat.Animation(convert(Int, floor(1.0 / Δt)))

#     T = length(q)
#     for t = 1:T-1
#         MeshCat.atframe(anim, t) do
#             # 현재 푸셔 위치 업데이트
#             _set_planar_push!(vis, model, q[t]; i=i)

#             # trail 추가 (과거 위치를 여러 개 남김)
#             for k = 1:trail_length
#                 if t-k > 0
#                     α = max(0.0, 0.5 - 0.1*(k-1))  # 투명도 점점 감소
#                     setobject!(vis["pusher_trail_$(i)_$(k)"],
#                         Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, r - r_pusher), r_pusher),
#                         MeshPhongMaterial(color = Colors.RGBA(0.5, 0.5, 0.5, α)))
#                     settransform!(vis["pusher_trail_$(i)_$(k)"], Translation(q[t-k][2], q[t-k][3], 0.01 * i))
#                 end
#             end
#         end
#     end

#     settransform!(vis["/Cameras/default"],
#         compose(Translation(0.0, 0.0, 50.0), LinearMap(RotZ(0.5 * pi) * RotY(-pi/2.5))))
#     setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 50)

#     MeshCat.setanimation!(vis, anim)
# end
