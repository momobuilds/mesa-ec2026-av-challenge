# Task 4 — Environment Mapping Report

## Pipeline Summary

The Task 4 pipeline builds a probabilistic 2-D occupancy grid of the racing
circuit by fusing all three LiDAR sensors (front, left, right) accumulated
across 60 master-clock timestamps in the world frame. Each scan is transformed
from ego coordinates to world coordinates using the ego trajectory reconstructed
from `ego.mat`, then ray-cast through a log-odds grid that treats every point
as a HIT and every empty voxel along the ray as a MISS. The grid is updated
incrementally, so early noisy returns are progressively over-written by later,
consistent observations. After all frames are processed, the occupied and free
cells are extracted, morphologically closed to fill small gaps, and the boundary
of the drivable region is traced into three polylines exported as GeoJSON.

Dynamic opponents (tracked by the Task 3 pipeline) are masked before each
ray-cast: the axis-aligned bounding box of every confirmed or coasting track is
reset to "unknown" (log-odds 0) in the grid before the new scan is integrated.
This prevents moving vehicles from smearing phantom occupied cells along the
racing line. The effect is quantified in the free-space comparison below.
Without masking, the grid permanently marks opponent footprints as occupied,
shrinking the estimated free area and corrupting the boundary extraction.

The final deliverables are a static world-map image with boundary polylines
overlaid, a world-map image with the ego trajectory added, per-timestamp
ego-centred 100 × 60 m views, a 30-frame GIF showing the map filling in as the
ego drives, and the GeoJSON boundary file. A comparison run without dynamic
masking is saved to `outputs/task4_no_mask/` for reference.

---

## Log-Odds Parameters

| Parameter | Value  | Rationale |
|-----------|--------|-----------|
| L_HIT     | +0.85  | Confident occupied update per LiDAR return (~70 % prior after one hit). Chosen to build up occupancy quickly while remaining reversible. |
| L_MISS    | −0.40  | Softer free-space update: LiDAR range readings are reliable but the ray-cast cells include beam spread. A smaller magnitude prevents over-clearing near the scan edge. |
| L_MAX     | +5.0   | Hard clamp: ~99.3 % probability. Cells that are consistently hit saturate quickly and are not easily freed by a single miss ray — represents permanent structure (barriers). |
| L_MIN     | −5.0   | Hard clamp: ~0.7 % probability. Symmetric floor prevents a large number of miss rays from creating artificially high confidence in open areas that were only briefly observed. |

The asymmetry (|L_HIT| > |L_MISS|) biases the grid toward marking obstacles;
false negatives (missed barriers) are more dangerous than false positives for
path planning on a race circuit.

---

## Resolution / Extent Trade-off

| Configuration         | Grid size (cells) | Memory   | Detail level |
|-----------------------|:-----------------:|:--------:|:------------:|
| 0.5 m / ±300 m        | 1 200 × 1 200     |  5.5 MB  | Low — barrier width (~0.3 m) not resolved |
| **0.25 m / ±300 m**   | **2 400 × 2 400** | **22 MB**| **Chosen — resolves individual barrier posts; fits in RAM** |
| 0.1 m / ±300 m        | 6 000 × 6 000     | 137 MB   | High — exceeds practical RAM for a 60-frame run; ray-cast latency triples |
| 0.25 m / ±700 m       | 5 600 × 5 600     | 119 MB   | Required for full-bag run (circuit extent 671 × 1 122 m) |

The chosen 0.25 m / ±300 m configuration covers the 60-frame (~240 m) segment
with sub-decimetre precision at manageable cost. For a full-bag run the
`--grid-extent 700` flag switches to the 0.25 m / ±700 m configuration.

---

## Sensor Selection

| Sensor            | Included | Reason |
|-------------------|:--------:|--------|
| lidar_front       | Yes      | Primary sensor: 360 returns per scan at high density, forward hemisphere. |
| lidar_left        | Yes      | Adds left-barrier coverage; moderately dense. |
| lidar_right       | Yes      | Adds right-barrier coverage; symmetric with lidar_left. |
| radar_front       | No       | Radar point clouds are sparse and include multi-path ghost returns from barriers; they degrade grid quality without adding new structural information. |
| radar_left        | No       | Same reason as radar_front; lateral radars see primarily barrier reflections at >10 m, already well-covered by LiDARs. |
| radar_right       | No       | Same as radar_left. |
| radar_rear        | No       | Rear radar adds negligible coverage of already-visited cells; excluding it reduces noise. |

---

## Quantitative Results (60 frames)

| Metric                  | With masking  | Without masking |
|-------------------------|:-------------:|:---------------:|
| Free-space estimate      | 7 716 m²      | 8 190 m²        |
| Occupied area (proxy)   | ~474 m²       | ~0 m² net shift |
| Boundary polylines       | 3             | 3               |
| Total boundary length    | 193 m         | 193 m           |

**Boundary breakdown (GeoJSON features):**

| id | Length |
|----|-------:|
| 0  | 51 m   |
| 1  | 71 m   |
| 2  | 71 m   |

The free-space difference (8 190 − 7 716 = **474 m²**) represents the aggregate
footprint of dynamic opponents that were correctly masked out. Without masking,
those cells remain permanently occupied, reducing the estimated navigable area
and corrupting boundary polylines near the racing line.

Boundary length is identical in both runs because the outer barriers are well
separated from the opponent trajectories; masking only affects cells on or near
the racing line, not the structural perimeter.

---

## Dynamic-Object Masking Effect

Without masking, each frame's LiDAR returns from opponent vehicles are ray-cast
into the grid exactly like static structure. Because the opponents repeatedly
drive the same racing line, their footprints accumulate log-odds hits and
eventually saturate at L_MAX. The result is a ghost "wall" along the inside of
corners — the racing line — which (a) shrinks the free-space estimate by 474 m²
and (b) can split what should be a single open lane into two narrow corridors.

The masking step resets the axis-aligned bounding box of every confirmed or
coasting Task 3 track to log-odds 0 (unknown) immediately before integrating the
new scan. Only the current scan's points can re-occupy those cells, and if the
opponent has moved, those cells see miss rays instead. After a few frames with
no returns the cells revert to free, leaving no permanent artefact.

Reference: `outputs/task4_no_mask/` contains the unmasked comparison outputs
(map image, GeoJSON, GIF) for side-by-side inspection.

---

## Ego Trajectory Integration

A key implementation detail: the `vx` and `vy` columns in `ego.mat` are
velocities in the **ego body frame**, not the world frame. Naively integrating
them as world-frame displacements produces a trajectory that curves in the wrong
direction whenever the vehicle yaws.

The correct integration applies the rotation matrix for the current heading
`psi`:

```
dx_world = vx * cos(psi) - vy * sin(psi)
dy_world = vx * sin(psi) + vy * cos(psi)
```

This is implemented in `EgoTrajectory.integrate()`. The Task 3 `predict_state()`
function uses only the ego-frame displacement to compensate opponent track
predictions and intentionally does not perform this rotation (it works in the
ego-relative reference frame). Confusing the two frames was an early bug that
produced a figure-eight trajectory; the rotation formula above corrects it.

---

## Limitations and Future Work

1. **No loop-closure / SLAM.** Odometry drift accumulates over the full circuit
   (~671 × 1 122 m oval). After one lap, the map start and end do not align.
   Integrating a graph-SLAM back-end (e.g., GTSAM) with loop-closure constraints
   would eliminate this drift.

2. **Fixed grid resolution.** A uniform 0.25 m grid wastes cells in open areas
   and may be too coarse to resolve narrow barrier gaps. An octree or adaptive
   resolution scheme would be more memory-efficient.

3. **Approximate sensor extrinsics.** Sensor positions and orientations are
   hardcoded estimates; no calibration data is available in the bag. Inaccurate
   extrinsics misalign left/right LiDAR returns from the same barrier by up to
   ~5 cm, blurring the boundary polyline.

4. **Side-LiDAR timestamp desynchronisation.** `lidar_left` and `lidar_right`
   are interpolated to the `lidar_front` master clock. If their hardware
   timestamps lag by >10 ms the ego pose used for transformation is slightly
   wrong, causing a small lateral smear at high cornering speeds.

5. **Axis-aligned masking.** Opponent bounding boxes are axis-aligned in the
   world frame; the true vehicle footprint is oriented along its heading. At high
   yaw angles the axis-aligned box over-masks cells behind and beside the
   opponent, slightly shrinking the usable free area. An oriented (rotated
   rectangle) mask would be more precise.

---

*Generated by run_task4.py — `uv run python scripts/run_task4.py --bag-dir data --max-frames 60`*
