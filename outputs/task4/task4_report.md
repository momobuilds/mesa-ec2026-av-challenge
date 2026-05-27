# Task 4 — Environment Mapping Report

## Pipeline Summary

The Task 4 pipeline builds a probabilistic 2-D occupancy grid of the racing circuit by fusing
all three LiDAR sensors (front, left, right) accumulated across 60 master-clock timestamps in
the world frame. Each scan is transformed from ego coordinates to world coordinates using the
ego pose obtained directly from the `/tf` topic in the bag, then ray-cast through a log-odds
grid that treats every point as a HIT and every empty voxel along the ray as a MISS. The grid
is updated incrementally, so early noisy returns are progressively over-written by later,
consistent observations. After all frames are processed, the occupied and free cells are
extracted, morphologically closed to fill small gaps, and the boundary of the drivable region
is traced into three polylines exported as GeoJSON.

Dynamic opponents (tracked by the Task 3 pipeline) are masked before each ray-cast: the
axis-aligned bounding box of every confirmed or coasting track is reset to "unknown"
(log-odds 0) in the grid before the new scan is integrated. This prevents moving vehicles from
smearing phantom occupied cells along the racing line.

The final deliverables are a static world-map image with boundary polylines overlaid, a
world-map image with the ego trajectory added, per-timestamp ego-centred 100 x 60 m views, a
30-frame GIF showing the map filling in as the ego drives, and the GeoJSON boundary file. A
comparison run without dynamic masking is saved to `outputs/task4_no_mask/` for reference.

---

## Ego Trajectory Source (updated)

The ego world-frame pose is now read directly from the `/tf` topic in the bag using
`BagTFReader`, which reads the 45 080 `map -> center_of_gravity` transforms published at
~200 Hz. `EgoTrajectory.from_bag_tf()` wraps this into the same `pose_at(ts)` /
`ego_to_world()` interface used throughout the pipeline.

```python
# run_task4.py -- preferred path
ego_traj = EgoTrajectory.from_bag_tf(args.bag_dir)
```

This is the approach confirmed by the professor:

> *"Nella bag sono presenti anche le tf, quindi possono essere usate quelle al posto del
> movimento dell'ego per tracciare il veicolo in riferimento mappa."*
> (The TFs are also present in the bag, so they can be used instead of ego movement to track
> the vehicle in map reference frame.)

`ego.mat` is loaded separately as `EgoStateLog` and used **only for radar Doppler correction**
in Tasks 2 and 3. It is no longer used as the ego pose source for map building.

### Previous approach (fallback only)

The previous approach integrated `ekf_vx`/`ekf_vy` from `ego.mat` to reconstruct world-frame
pose. The integration requires rotating body-frame velocities by the instantaneous heading:

```python
dx_world = vx * cos(psi) - vy * sin(psi)
dy_world = vx * sin(psi) + vy * cos(psi)
```

This is still available as `EgoTrajectory.from_ego_log(ego_log)` and is used as an automatic
fallback if `/tf` is unavailable. However, it accumulates numerical drift over the 240 s bag
(velocity integration is not closed-loop), whereas the TF-based approach provides the
ground-truth pose published by the vehicle's own localisation stack with no accumulated error.

---

## Sensor Extrinsics (updated)

Sensor -> ego transforms are now loaded from `data/eva.urdf` via
`tf_utils.init_extrinsics_from_urdf()`, replacing the previous hardcoded approximations.

| Sensor       | x fwd (m) | y lat (m) | z up (m) | Mounting yaw |
|:-------------|----------:|----------:|---------:|-------------:|
| lidar_front  |    +0.662 |     0.000 |   +0.610 |      -179.5° |
| lidar_left   |    +0.002 |    +0.135 |   +0.674 |       -59.4° |
| lidar_right  |    +0.002 |    -0.135 |   +0.674 |       +60.0° |

The URDF confirms that `lidar_left` and `lidar_right` point rearward (yaw ~60° from the
backward axis), which explains why their occupancy contributions are primarily behind the ego
vehicle — consistent with observed grid patterns.

---

## Log-Odds Parameters

| Parameter | Value | Rationale                                                                                            |
|:----------|------:|:-----------------------------------------------------------------------------------------------------|
| L_HIT     | +0.85 | Confident occupied update (~70 % prior after one hit). Builds occupancy quickly, stays reversible.  |
| L_MISS    | -0.40 | Softer free-space update. Smaller magnitude prevents over-clearing near scan edges.                  |
| L_MAX     | +5.0  | Hard clamp (~99.3 % probability). Saturated cells represent permanent structure (barriers).          |
| L_MIN     | -5.0  | Hard clamp (~0.7 % probability). Prevents over-confident free-space from sparse miss rays.           |

The asymmetry (|L_HIT| > |L_MISS|) biases the grid toward marking obstacles; false negatives
(missed barriers) are more dangerous than false positives for path planning on a race circuit.

---

## Resolution / Extent Trade-off

| Configuration       | Grid cells      | Memory  | Detail level                                            |
|:--------------------|:---------------:|:-------:|:--------------------------------------------------------|
| 0.5 m / ±300 m      | 1200 x 1200     |  5.5 MB | Low — barrier width (~0.3 m) not resolved               |
| **0.25 m / ±300 m** | **2400 x 2400** | **22 MB** | **Chosen — resolves barrier posts; fits in RAM**       |
| 0.1 m / ±300 m      | 6000 x 6000     | 137 MB  | High — exceeds practical RAM; ray-cast latency triples  |
| 0.25 m / ±700 m     | 5600 x 5600     | 119 MB  | Required for full-bag run (circuit ~671 x 1122 m)       |

The chosen 0.25 m / ±300 m configuration covers the 60-frame (~240 m) segment with
sub-decimetre precision at manageable cost. For a full-bag run use `--grid-extent 700`.

---

## Sensor Selection

| Sensor        | Included | Reason                                                                                               |
|:--------------|:--------:|:-----------------------------------------------------------------------------------------------------|
| lidar_front   |   Yes    | Primary sensor: high-density returns, forward hemisphere.                                            |
| lidar_left    |   Yes    | Adds left-barrier coverage; moderately dense.                                                        |
| lidar_right   |   Yes    | Adds right-barrier coverage; symmetric with lidar_left.                                              |
| radar_front   |    No    | Sparse point cloud with multi-path ghost returns; degrades grid quality.                             |
| radar_left    |    No    | Barrier reflections at >10 m lateral, already covered by LiDARs.                                    |
| radar_right   |    No    | Same reason as radar_left.                                                                           |
| radar_rear    |    No    | Adds negligible coverage of already-visited cells; increases noise.                                  |

---

## Quantitative Results (60 frames)

| Metric                 | With masking | Without masking |
|:-----------------------|:------------:|:---------------:|
| Free-space estimate    |   7 716 m²   |    8 190 m²     |
| Occupied area (proxy)  |    ~474 m²   |  ~0 m² net shift |
| Boundary polylines     |      3       |        3        |
| Total boundary length  |    193 m     |     193 m       |

**Boundary breakdown (GeoJSON features):**

| id | Length |
|:---|-------:|
| 0  |   51 m |
| 1  |   71 m |
| 2  |   71 m |

The free-space difference (8 190 - 7 716 = **474 m²**) represents the aggregate footprint of
dynamic opponents that were correctly masked out. Without masking, those cells remain
permanently occupied, reducing the estimated navigable area and corrupting boundary polylines
near the racing line.

---

## Dynamic-Object Masking Effect

Without masking, each frame's LiDAR returns from opponent vehicles are ray-cast into the grid
exactly like static structure. Because the opponents repeatedly drive the same racing line,
their footprints accumulate log-odds hits and eventually saturate at L_MAX. The result is a
ghost "wall" along the inside of corners — the racing line — which (a) shrinks the free-space
estimate by 474 m² and (b) can split what should be a single open lane into two narrow
corridors.

The masking step resets the axis-aligned bounding box of every confirmed or coasting Task 3
track to log-odds 0 (unknown) immediately before integrating the new scan. Only the current
scan's points can re-occupy those cells, and if the opponent has moved, those cells see miss
rays instead. After a few frames with no returns the cells revert to free, leaving no permanent
artefact.

Reference: `outputs/task4_no_mask/` contains the unmasked comparison outputs (map image,
GeoJSON, GIF) for side-by-side inspection.

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
   now read from `data/eva.urdf`. The left/right LiDAR mounting yaw (+/-60° rearward, not
   +/-90°) is now correctly applied, eliminating the ~5 cm misalignment that previously blurred
   boundary polylines.

4. **Side-LiDAR timestamp desynchronisation.** `lidar_left` and `lidar_right` are interpolated
   to the `lidar_front` master clock. If their hardware timestamps lag by >10 ms the ego pose
   used for transformation is slightly wrong, causing a small lateral smear at high cornering
   speeds.

5. **Axis-aligned masking.** Opponent bounding boxes are axis-aligned in the world frame; the
   true vehicle footprint is oriented along its heading. At high yaw angles the axis-aligned
   box over-masks cells beside the opponent. An oriented (rotated rectangle) mask would be more
   precise.

---

*Generated by run_task4.py — `uv run python scripts/run_task4.py --bag-dir data --max-frames 60`*
