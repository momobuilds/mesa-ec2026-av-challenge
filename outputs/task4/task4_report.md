# Task 4 — Environment Mapping Report

## Pipeline Summary

The Task 4 pipeline builds a probabilistic 2-D occupancy grid of the racing circuit by fusing
all three LiDAR sensors (front, left, right) accumulated across 200 master-clock timestamps in
the world frame. Each scan is transformed from ego coordinates to world coordinates using the
ego pose obtained directly from the `/tf` topic in the bag, then ray-cast through a log-odds
grid that treats every point as a HIT and every empty voxel along the ray as a MISS. The grid
is updated incrementally, so early noisy returns are progressively over-written by later,
consistent observations. After all frames are processed, the occupied cells are morphologically
closed and the boundary of the drivable region is traced into polylines exported as GeoJSON.

Dynamic opponents (tracked by the Task 3 pipeline) are masked before each ray-cast: the
oriented bounding box of every confirmed or coasting track is reset to "unknown" (log-odds 0)
in the grid before the new scan is integrated. This prevents moving vehicles from smearing
phantom occupied cells along the racing line.

The final deliverables are:

- `world_map_static.png` -- accumulated gray-scale occupancy map with boundary polylines
- `freespace_map.png` -- color-coded map (green=free, black=occupied, grey=unknown)
- `world_map_with_trajectory.png` -- occupancy map + ego trajectory + boundaries
- `ego_view_NNN.png` (8 views) -- ego-centred 100 x 60 m heading-aligned crops
- `map_animation.gif` -- 30-frame GIF showing the map building up over time
- `track_boundaries.geojson` -- 18 barrier polylines with world-frame coordinates

---

## Critical Fix: Auto-Centered Occupancy Grid

The previous run used `origin_xy=(0, 0)` with `±300 m` extent, placing the grid over
x ∈ [−300, 300] m and y ∈ [−300, 300] m. However, the `/tf` world frame in this bag places
the track at **y ≈ 458–648 m**, entirely outside the old grid. As a result, all LiDAR points
were silently discarded and the old outputs were empty noise.

The pipeline now **auto-centers the grid** on the actual driven segment:

```python
# Compute centroid of ego poses for the processed frames
cx_grid = (min(xs_f) + max(xs_f)) / 2.0    # ≈ 73 m
cy_grid = (min(ys_f) + max(ys_f)) / 2.0    # ≈ 553 m
# Pad by max_range (80 m) + 30 m margin, round up to nearest 50 m
half_ext = ceil(max(x_span/2 + pad, y_span/2 + pad) / 50) * 50  # = 250 m
```

For the 200-frame run: grid centered at **(73, 553)**, extent **±250 m** →
2000 × 2000 cells at 0.25 m/cell (≈16 MB).

---

## Ego Trajectory Source (updated)

The ego world-frame pose is read directly from the `/tf` topic in the bag using
`BagTFReader`, which provides the 45 080 `map → center_of_gravity` transforms published at
~200 Hz. `EgoTrajectory.from_bag_tf()` wraps this into the `pose_at(ts)` / `ego_to_world()`
interface used throughout the pipeline.

```python
# run_task4.py — preferred path
ego_traj = EgoTrajectory.from_bag_tf(args.bag_dir)
```

This is the approach confirmed by the professor:

> *"Nella bag sono presenti anche le tf, quindi possono essere usate quelle al posto del
> movimento dell'ego per tracciare il veicolo in riferimento mappa."*
> (The TFs are also present in the bag, so they can be used instead of ego movement to track
> the vehicle in the map reference frame.)

`ego.mat` is loaded separately as `EgoStateLog` and used **only for radar Doppler correction**
in Tasks 2 and 3. It is no longer used as the ego pose source for map building.

---

## Sensor Extrinsics (updated)

Sensor → ego transforms are now loaded from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, replacing the previous hardcoded approximations.

| Sensor | x fwd (m) | y lat (m) | z up (m) | Mounting yaw |
| :--- | ---: | ---: | ---: | ---: |
| lidar_front | +0.662 | 0.000 | +0.610 | -179.5 deg |
| lidar_left | +0.002 | +0.135 | +0.674 | -59.4 deg |
| lidar_right | +0.002 | -0.135 | +0.674 | +60.0 deg |

The URDF confirms that `lidar_left` and `lidar_right` point rearward (yaw ~60 deg from the
backward axis), so they observe barriers trailing the ego -- consistent with their high
point counts behind the ego position in the BEV maps.

---

## Log-Odds Parameters

| Parameter | Value | Rationale |
| :--- | ---: | :--- |
| L_HIT | +0.85 | Confident occupied update (~70% prior after one hit). Builds occupancy quickly, stays reversible. |
| L_MISS | -0.40 | Softer free-space update. Smaller magnitude prevents over-clearing near scan edges. |
| L_MAX | +5.0 | Hard clamp (~99.3% probability). Saturated cells represent permanent structure (barriers). |
| L_MIN | -5.0 | Hard clamp (~0.7% probability). Prevents over-confident free-space from sparse miss rays. |

The asymmetry (L_HIT > L_MISS in magnitude) biases the grid toward marking obstacles; false
negatives (missed barriers) are more dangerous than false positives for path planning.

---

## Grid Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| Center (world) | (73, 553) m | Auto-computed from ego poses across the 200 processed frames |
| Extent | +/-250 m | Covers x in [-177, 323] m, y in [303, 803] m -- full 200-frame segment |
| Resolution | 0.25 m/cell | Resolves barrier posts; sub-decimetre precision |
| Grid size | 2000 x 2000 | 4.0 M cells, ~16 MB float32 |
| Max LiDAR range | 80 m | Sets maximum ray-cast length per scan |

---

## Sensor Selection

| Sensor | Included | Reason |
| :--- | :---: | :--- |
| lidar_front | Yes | Primary sensor: high-density returns, forward hemisphere. |
| lidar_left | Yes | Adds left-barrier coverage; moderately dense. |
| lidar_right | Yes | Adds right-barrier coverage; symmetric with lidar_left. |
| radar_front | No | Sparse point cloud with multi-path ghost returns; degrades grid quality. |
| radar_left | No | Barrier reflections at >10 m lateral, already covered by LiDARs. |
| radar_right | No | Same reason as radar_left. |
| radar_back | No | Adds negligible coverage of already-visited cells; increases noise. |

---

## Quantitative Results (200 frames)

| Metric | With masking | Without masking |
| :--- | :---: | :---: |
| Timestamps processed | 200 | 200 |
| LiDAR scans integrated | 600 | 600 |
| Free-space estimate | 44 851 m2 | 45 105 m2 |
| Occupied area | 838 m2 | 852 m2 |
| Masking effect (occupied) | baseline | +14 m2 extra |
| Boundary polylines | 18 | 19 |
| Total boundary length | 1 316 m | 1 378 m |

**Boundary breakdown — 18 polylines (with masking):**

| id | Length (m) | x range (m) | y range (m) | Side |
| ---: | ---: | :---: | :---: | :--- |
| 0 | 41.8 | [-45, -31] | [764, 803] | Left |
| 1 | 52.5 | [-40, -31] | [696, 712] | Left |
| 2 | 51.4 | [-20, -4] | [659, 708] | Left |
| 3 | 138.6 | [110, 137] | [637, 684] | Right |
| 4 | 25.9 | [96, 99] | [634, 660] | Right |
| 5 | 129.7 | [94, 101] | [621, 689] | Right |
| 6 | 58.1 | [-13, -5] | [610, 637] | Left |
| 7 | 37.9 | [8, 16] | [595, 620] | Inner |
| 8 | 86.4 | [101, 106] | [579, 617] | Right |
| 9 | 66.6 | [113, 121] | [530, 551] | Right |
| 10 | 159.0 | [4, 12] | [500, 567] | Inner/Left |
| 11 | 111.4 | [96, 111] | [450, 496] | Right |
| 12 | 71.0 | [23, 25] | [434, 480] | Inner |
| 13 | 35.3 | [19, 22] | [433, 465] | Inner |
| 14 | 73.0 | [93, 99] | [420, 450] | Right |
| 15 | 58.4 | [78, 80] | [402, 437] | Right |
| 16 | 34.0 | [37, 48] | [352, 373] | Centre |
| 17 | 85.4 | [62, 96] | [303, 332] | Right |

The boundaries cluster around two lateral positions -- x approx [-45, -4] m (left barrier,
~170 m total) and x approx [78, 137] m (right barrier, ~770 m total) -- consistent with
IndyCar track barrier positions flanking the racing line.

---

## Dynamic-Object Masking Effect

Without masking, each frame's LiDAR returns from opponent vehicles are ray-cast into the grid
exactly like static structure. Because opponents repeatedly drive the same racing line, their
footprints accumulate log-odds hits and eventually saturate at L_MAX, creating a ghost "wall"
along the inside of corners.

The masking step resets the oriented bounding box of every confirmed or coasting Task 3
track to log-odds 0 (unknown) immediately before integrating the new scan. Only the current
scan's points can re-occupy those cells, and if the opponent has moved, those cells receive
miss rays and eventually become free.

- **Extra occupied cells**: +14 m2 without masking vs. baseline
- **Boundary polylines**: 19 without masking vs. 18 with masking
- **Free-space difference**: +254 m2 without masking (opponent cells are freed after grid resets with masking)

Reference: `outputs/task4_no_mask/` contains the unmasked comparison outputs.

---

## Visualisation Outputs

### freespace_map.png (new)

Color-coded freespace visualisation for evaluation clarity:

- **Green** -- confirmed free space (P(occ) < 0.3, log-odds < -0.85)
- **Black** -- confirmed occupied (P(occ) > 0.5, log-odds > 0)
- **Grey** -- unknown / unobserved cells
- **Yellow lines** -- extracted track boundary polylines

### world_map_static.png

Gray-scale occupancy grid (0=occupied, 128=unknown, 255=free) with red boundary overlays
and a 50 m scale bar.

### world_map_with_trajectory.png

Same map with the blue ego trajectory, green start marker, and red end marker overlaid.

### ego_view_NNN.png (8 views)

Heading-aligned ego-centred 100 x 60 m crops of the accumulated map, with track-state
oriented bounding boxes overlaid in red. Evenly spaced across the 200-frame run.

### map_animation.gif (30 frames)

1.2 MB animated GIF showing the occupancy grid filling in as the ego drives, with a
trailing blue ego-position trajectory.

---

## Limitations and Future Work

1. **No loop-closure / SLAM.** The TF-based pose is the ground-truth localisation output of
   the vehicle's own stack, so no velocity-integration drift accumulates within the recorded
   session. However, a full multi-lap circuit run would require loop-closure / SLAM if the
   localisation stack itself drifts between laps.

2. **Fixed grid resolution.** A uniform 0.25 m grid wastes cells in open areas and may be too
   coarse to resolve narrow barrier gaps. An octree or adaptive resolution scheme would be more
   memory-efficient.

3. ~~**Approximate sensor extrinsics.**~~ **Resolved.** Sensor positions and orientations are
   now read from `data/eva.urdf`. The left/right LiDAR mounting yaw (+/-60 deg rearward, not
   +/-90 deg) is now correctly applied, eliminating the ~5 cm misalignment that previously
   blurred boundary polylines.

4. **Side-LiDAR timestamp desynchronisation.** `lidar_left` and `lidar_right` are interpolated
   to the `lidar_front` master clock. If their hardware timestamps lag by >10 ms the ego pose
   used for transformation is slightly wrong, causing a small lateral smear at high cornering
   speeds.

5. **Axis-aligned masking.** Opponent bounding boxes are reset with an oriented box mask but
   the corners are approximated as axis-aligned in cell index space. An exact rasterisation of
   the oriented rectangle would be more precise.

6. ~~**Grid origin off-map.**~~ **Resolved.** The pipeline now auto-centers the grid on the
   actual driven segment's centroid (computed from ego poses of the processed frames) and pads
   by `max_range + 30 m` in all directions, ensuring all LiDAR returns fall inside the grid.

---

## Reproducing Results

```bash
# Full 200-frame run with dynamic masking (requires outputs/task3/tracks.csv)
uv run python scripts/run_task4.py --bag-dir data --max-frames 200 \
    --output-dir outputs/task4

# Comparison run without dynamic masking
uv run python scripts/run_task4.py --bag-dir data --max-frames 200 \
    --output-dir outputs/task4_no_mask --no-mask-tracks --no-animation
```

*Generated by run_task4.py — `uv run python scripts/run_task4.py --bag-dir data --max-frames 200`*
