# Point vs Line Contact: Current Setup Report

## 1. Scope
This report summarizes the **current code settings** for:
- Point contact model (`planarpush` / `examples/point_box_moving.jl`)
- Line contact model (`lineplanarpush_xy` / `examples/line_box_moving.jl`)
- Comparison and robustness test scripts

All values below are taken from the current repository state.

## 2. Core Model Parameter Comparison

| Item | Point Contact | Line Contact |
|---|---:|---:|
| Model file | `src/models/planar_push/model.jl` | `src/models/line_planar_push_xy/model.jl` |
| `μ_surface` | `0.5` | `1.0` |
| `μ_pusher` | `0.5` | `0.5` |
| `mass_block` | `1.0` | `1.0` |
| `mass_pusher` | `10.0` | `5.0` (per pusher point) |
| `nw` (model disturbance dim) | implicit `0` in planar push setup | `1` (`nw = 1`) |
| Spring-damper | none | `L=0.05`, `k_spring=300`, `c_damping=50` |

Notes:
- The line model includes an internal spring-damper coupling between two pusher points.
- In line residual dynamics, disturbance enters as `[0; 0; w1; 0; 0; 0; 0]` when `nw > 0`.

## 3. Trajectory Optimization Test Condition Comparison

### 3.1 Shared high-level settings
- Horizon step: `h = 0.05`
- Horizon length: `T = 26`
- Goal: `x_goal=0.3`, `y_goal=0.2`, `θ_goal=0.8`
- Cost structure includes state/velocity/control and pusher movement penalty (`Wp_move`)
- Solver options: `max_iter=10`, `max_al_iter=20`, `con_tol=0.005`

### 3.2 State/control dimensions and bounds

| Item | Point | Line |
|---|---:|---:|
| `nq` | 5 | 7 |
| `nu` | 2 | 4 |
| Control lower bound `ul` | `[-5.0, -5.0]` | `[0.0, -2.5, 0.0, -2.5]` |
| Control upper bound `uu` | `[5.0, 5.0]` | `[2.5, 2.5, 2.5, 2.5]` |

### 3.3 Contact/slip constraint alignment (currently matched)

Both scripts are now aligned to:
- `max_pusher_gap = 0.0001`
- `max_tangent_slip = 0.005`

Implementation:
- Point: one slip channel in block-local tangent direction.
- Line: two slip channels (`slip1`, `slip2`) for two pusher points.

## 4. Comparison Script (`examples/compare_point_line.jl`)

### 4.1 What it compares
- Final position error norm
- Final orientation error (abs)
- `control_effort = sum(u_t' * u_t)`
- `total_u_mag = sum_t ||u_t||`
- Contact force stats (`gamma_mean_abs`, `gamma_peak`)
- Slip max vs slip bound

### 4.2 Output files
Saved to `planar_push_data/`:
- `compare_point_line_theta.png`
- `compare_point_line_u_norm.png`
- `compare_point_line_u_channels.png`
- `compare_point_line_u_cumsum.png`
- `compare_point_line_u_total_bar.png`
- `compare_point_line_slip.png`

## 5. Robustness Monte Carlo Script (`examples/compare_point_line_robustness.jl`)

### 5.1 MC settings
- `N_MC = 50`
- Disturbance std per step:
  - `DIST_STD_X = 0.0015`
  - `DIST_STD_Y = 0.0015`
  - `DIST_STD_THETA = 0.004`
- Success criteria:
  - `norm(final position error) <= 0.03`
  - `abs(final theta error) <= 0.10`
- RNG seed: `MersenneTwister(1234)`

### 5.2 What is tested
For both point and line models, under shared random disturbance trials:
- Open-loop rollout (nominal controls only)
- Online feedback rollout (`u = u_nom + K_t(x - x_nom)`, with input clamping)

### 5.3 Robustness outputs
Saved to `planar_push_data/`:
- Mean bars:
  - `robustness_pos_mean_bar.png`
  - `robustness_theta_mean_bar.png`
  - `robustness_total_u_mean_bar.png`
  - `robustness_success_rate_bar.png`
- Distribution histograms:
  - `robustness_pos_hist.png`
  - `robustness_theta_hist.png`
  - `robustness_total_u_hist.png`

## 6. MeshCat Visual Comparison Script

Script: `examples/visualize_point_line_nominal_open_feedback.jl`

Current visualization behavior:
- One MeshCat server per model overlaying three trajectories:
  - blue: nominal
  - red: disturbed open-loop
  - green: disturbed online-feedback
- Goal box shown as a solid red box.
- Camera fixed for both point and line overlay views.

## 7. Repro Commands

```bash
julia --project=. examples/compare_point_line.jl
julia --project=. examples/compare_point_line_robustness.jl
julia --project=. examples/visualize_point_line_nominal_open_feedback.jl
```

## 8. Important Interpretation Notes
- Point vs line are **not fully identical physical models** (`μ_surface`, pusher mass, spring-damper, disturbance dimension differ).
- Constraint thresholds (`max_pusher_gap`, `max_tangent_slip`) are currently matched.
- Robustness script uses the same disturbance trial set across methods for fair comparison.
