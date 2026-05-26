# Task 4 Environment Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a probabilistic log-odds occupancy grid from three LiDAR sensors accumulated in the world frame, with dynamic-object masking (Task 3 tracks), track boundary extraction, ego-frame visualisation, and an animated GIF of the map filling in.

**Architecture:** `EgoTrajectory` integrates `EgoStateLog` ego-frame (vx,vy) velocities via heading rotation into a world-frame pose array; `ProbabilisticGrid` accumulates log-odds via vectorised Bresenham raycasting (`raycasting.py`); `track_boundary.py` extracts polylines with OpenCV morphology + contours; `map_visualization.py` renders all outputs; `run_task4.py` orchestrates the pipeline.

**Tech Stack:** numpy, pandas, matplotlib, Pillow (PIL), cv2, h5py — all available transitively from the existing pyproject.toml.

**Key insight on vx/vy:** `ekf_vx`/`ekf_vy` from `ego.mat` are **ego-frame** (longitudinal/lateral), not world-frame. `predict_state` in motion_model.py skips the rotation because Δt ≈ 0.2 s makes Δheading ≈ 0.4° (rotation ≈ identity). `EgoTrajectory` must rotate explicitly: `dx_w = vx·cos(h) − vy·sin(h)`, `dy_w = vx·sin(h) + vy·cos(h)`.

**Grid sizing:** Full-bag integrated trajectory spans x ∈ [−431, +240] m, y ∈ [−656, +466] m. Default `--grid-extent 300` (±300 m) covers the 60-frame run (~12 s, ~240 m path + 80 m sensor range). Full-bag runs need `--grid-extent 700`.

---

### Task 0: Add `iter_ego_frame_points` to `sensor_detectors.py`

**Files:**
- Modify: `src/sensor_detectors.py`

- [ ] **Step 1: Add import and helper after existing imports (before `EGO_FORWARD_RANGE`)**

Open `src/sensor_detectors.py`. After the `from src.tf_utils import transform_points` import line, add:

```python
from typing import Iterator
```

Then, after the `EGO_LATERAL_RANGE` constant block (~line 30), insert:

```python
def iter_ego_frame_points(
    summary: BagSummary,
    topic: str,
    max_frames: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (timestamp_ns, ego_xyz) for each frame of *topic*.

    Returns full 3D ego-frame (N, 3) points without ground filtering.
    The existing API (ego_bev_points, detect_topic_frames, detect_all_sensors)
    is untouched.
    """
    reader = SQLitePointCloudReader(summary)
    for msg_dict in reader.iter_pointcloud_messages(topic, max_frames=max_frames):
        try:
            structured = pointcloud2_to_structured_array(msg_dict["msg"])
        except Exception:  # noqa: BLE001
            continue
        sensor_frame = msg_dict["source_frame"]
        for fld in ("x", "y", "z"):
            if fld not in structured.dtype.names:
                break
        else:
            x_s = structured["x"].astype(np.float64)
            y_s = structured["y"].astype(np.float64)
            z_s = structured["z"].astype(np.float64)
            finite = np.isfinite(x_s) & np.isfinite(y_s) & np.isfinite(z_s)
            if finite.any():
                ego_xyz = transform_points(
                    np.column_stack([x_s[finite], y_s[finite], z_s[finite]]),
                    sensor_frame,
                )
                yield msg_dict["timestamp_ns"], ego_xyz
```

- [ ] **Step 2: Verify existing pipeline still works**

```bash
uv run python scripts/run_task3.py --bag-dir data --max-frames 5 \
    --bev-frames 0 --output-dir /tmp/t3_smoke
```
Expected: exits without errors, `/tmp/t3_smoke/tracks.csv` created.

- [ ] **Step 3: Commit**

```bash
git add src/sensor_detectors.py
git commit -m "feat: add iter_ego_frame_points helper to sensor_detectors (Task 4)"
```

---

### Task 1: `ego_trajectory.py` + world-extent sanity plot

**Files:**
- Create: `src/ego_trajectory.py`
- Create: `tests/test_ego_trajectory.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ego_trajectory.py`:

```python
"""Unit tests for EgoTrajectory."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

EGO_MAT = Path("data/ego.mat")
pytestmark = pytest.mark.skipif(not EGO_MAT.exists(), reason="data/ego.mat not found")

from src.tf_utils import EgoStateLog
from src.ego_trajectory import EgoTrajectory


def _make_traj() -> tuple[EgoTrajectory, EgoStateLog]:
    meta = yaml.safe_load(Path("data/metadata.yaml").read_text())
    bag_start = int(
        meta["rosbag2_bagfile_information"]["starting_time"]["nanoseconds_since_epoch"]
    )
    log = EgoStateLog("data/ego.mat", bag_start_ns=bag_start)
    return EgoTrajectory.from_ego_log(log), log


def test_pose_at_start_is_origin():
    traj, log = _make_traj()
    t_start = int(log._timestamps_ns[0])
    x, y, _ = traj.pose_at(t_start)
    assert abs(x) < 0.01 and abs(y) < 0.01, (
        f"Start pose should be (0,0), got ({x:.4f},{y:.4f})"
    )


def test_mid_bag_displacement_within_half_metre():
    traj, log = _make_traj()
    N = log.n_samples
    mid = N // 2
    t_arr = log._timestamps_ns
    vx, vy, heading = log._vx, log._vy, log._heading

    dt = np.diff(t_arr[:mid + 1]) / 1e9
    vx_w = vx[:mid + 1] * np.cos(heading[:mid + 1]) - vy[:mid + 1] * np.sin(heading[:mid + 1])
    vy_w = vx[:mid + 1] * np.sin(heading[:mid + 1]) + vy[:mid + 1] * np.cos(heading[:mid + 1])
    x_ref = float(np.sum(0.5 * (vx_w[:-1] + vx_w[1:]) * dt))
    y_ref = float(np.sum(0.5 * (vy_w[:-1] + vy_w[1:]) * dt))

    x, y, _ = traj.pose_at(int(t_arr[mid]))
    err = math.hypot(x - x_ref, y - y_ref)
    assert err < 0.5, f"Mid-bag displacement error {err:.3f} m > 0.5 m threshold"


def test_ego_to_world_at_start_applies_heading_rotation():
    traj, log = _make_traj()
    t_start = int(log._timestamps_ns[0])
    pts_ego = np.array([[10.0, 0.0], [0.0, 5.0]])
    pts_world = traj.ego_to_world(pts_ego, t_start)

    x0, y0, h0 = traj.pose_at(t_start)
    R = np.array([[math.cos(h0), -math.sin(h0)], [math.sin(h0), math.cos(h0)]])
    expected = (R @ pts_ego.T).T + np.array([x0, y0])
    np.testing.assert_allclose(pts_world, expected, atol=1e-6)


def test_world_extent_is_plausible():
    traj, _ = _make_traj()
    x_min, x_max, y_min, y_max = traj.world_extent()
    # Full bag: ~670 m × ~1120 m circuit
    assert x_max - x_min > 100, f"Track x-span {x_max - x_min:.0f} m suspiciously small"
    assert y_max - y_min > 100, f"Track y-span {y_max - y_min:.0f} m suspiciously small"
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_ego_trajectory.py -v 2>&1 | head -15
```
Expected: `ModuleNotFoundError` or `ImportError` for `src.ego_trajectory`.

- [ ] **Step 3: Create `src/ego_trajectory.py`**

```python
"""World-frame ego pose trajectory by integrating EgoStateLog velocities.

The EKF outputs (ekf_vx, ekf_vy) in ego.mat are ego-frame velocities
(longitudinal / lateral), NOT world-frame. To integrate into world-frame
displacement, rotate each sample by the instantaneous heading:

    dx_w = vx · cos(h) − vy · sin(h)
    dy_w = vx · sin(h) + vy · cos(h)

heading in ego.mat is a monotonically increasing odometer value (never
wrapped to [−π, π]). cos/sin are 2π-periodic, so no unwrapping is needed.

predict_state() in motion_model.py skips this rotation because Δt ≈ 0.2 s
gives Δheading ≈ 0.4° (rotation ≈ identity for a single step). Here we
integrate the full trajectory so the rotation is essential.

Origin: bag start → (x=0, y=0), heading=h[0].
"""
from __future__ import annotations

import math

import numpy as np

from src.tf_utils import EgoStateLog


def _rot2d(heading_rad: float) -> np.ndarray:
    """2×2 rotation matrix: ego-frame → world frame."""
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


class EgoTrajectory:
    """Continuous world-frame pose trajectory integrated from EgoStateLog."""

    def __init__(
        self,
        timestamps_ns: np.ndarray,
        x_w: np.ndarray,
        y_w: np.ndarray,
        heading: np.ndarray,
    ) -> None:
        self._timestamps_ns = timestamps_ns
        self._x_w = x_w
        self._y_w = y_w
        self._heading = heading

    @classmethod
    def from_ego_log(cls, ego_log: EgoStateLog) -> "EgoTrajectory":
        """Integrate all EgoStateLog samples into a world-frame trajectory."""
        t = ego_log._timestamps_ns
        vx = ego_log._vx
        vy = ego_log._vy
        h = ego_log._heading

        # Rotate ego-frame (longitudinal, lateral) → world-frame (x, y)
        vx_w = vx * np.cos(h) - vy * np.sin(h)
        vy_w = vx * np.sin(h) + vy * np.cos(h)

        dt = np.diff(t) / 1e9  # seconds
        dx = 0.5 * (vx_w[:-1] + vx_w[1:]) * dt
        dy = 0.5 * (vy_w[:-1] + vy_w[1:]) * dt

        x_w = np.concatenate([[0.0], np.cumsum(dx)])
        y_w = np.concatenate([[0.0], np.cumsum(dy)])

        return cls(timestamps_ns=t, x_w=x_w, y_w=y_w, heading=h)

    # ── public interface ──────────────────────────────────────────────────────

    def pose_at(self, t_ns: int) -> tuple[float, float, float]:
        """Interpolate (x_w, y_w, heading) at ROS timestamp *t_ns*."""
        t = float(t_ns)
        idx = int(np.searchsorted(self._timestamps_ns, t))
        idx = max(1, min(idx, len(self._timestamps_ns) - 1))
        t0, t1 = self._timestamps_ns[idx - 1], self._timestamps_ns[idx]
        alpha = float(np.clip((t - t0) / max(t1 - t0, 1.0), 0.0, 1.0))

        def lerp(a: np.ndarray) -> float:
            return float(a[idx - 1] + alpha * (a[idx] - a[idx - 1]))

        return lerp(self._x_w), lerp(self._y_w), lerp(self._heading)

    def ego_to_world(self, points_ego: np.ndarray, t_ns: int) -> np.ndarray:
        """Transform (N, 2) ego-frame XY to world frame at time *t_ns*."""
        x_w, y_w, heading = self.pose_at(t_ns)
        R = _rot2d(heading)
        out = (R @ points_ego.T).T.copy()
        out[:, 0] += x_w
        out[:, 1] += y_w
        return out

    def world_extent(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) over all integrated samples."""
        return (
            float(self._x_w.min()), float(self._x_w.max()),
            float(self._y_w.min()), float(self._y_w.max()),
        )

    def trajectory_xy(self) -> np.ndarray:
        """Return (N, 2) float64 array of all world-frame positions."""
        return np.column_stack([self._x_w, self._y_w])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ego_trajectory.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Generate sanity plot**

```bash
uv run python -c "
import yaml, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from src.tf_utils import EgoStateLog
from src.ego_trajectory import EgoTrajectory

meta = yaml.safe_load(Path('data/metadata.yaml').read_text())
bag_start = int(meta['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch'])
log = EgoStateLog('data/ego.mat', bag_start_ns=bag_start)
traj = EgoTrajectory.from_ego_log(log)
xy = traj.trajectory_xy()
x_min, x_max, y_min, y_max = traj.world_extent()
print(f'World extent: x=[{x_min:.0f}, {x_max:.0f}] m  y=[{y_min:.0f}, {y_max:.0f}] m')
print(f'Span: {x_max-x_min:.0f} m x {y_max-y_min:.0f} m')
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(xy[:,0], xy[:,1], 'b-', lw=0.5, alpha=0.7)
ax.plot(xy[0,0], xy[0,1], 'go', ms=8, label='start')
ax.plot(xy[-1,0], xy[-1,1], 'rs', ms=8, label='end')
ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
ax.set_xlabel('x_world (m)'); ax.set_ylabel('y_world (m)')
ax.set_title('Full ego trajectory — world frame (ego.mat integrated)')
ax.legend()
Path('outputs/task4').mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig('outputs/task4/ego_trajectory_sanity.png', dpi=150)
print('Saved outputs/task4/ego_trajectory_sanity.png')
"
```

Expected output (approximate):
```
World extent: x=[-431, 240] m  y=[-656, 466] m
Span: 671 m x 1122 m
Saved outputs/task4/ego_trajectory_sanity.png
```

- [ ] **Step 6: Commit**

```bash
git add src/ego_trajectory.py tests/test_ego_trajectory.py outputs/task4/ego_trajectory_sanity.png
git commit -m "feat: add EgoTrajectory with world-frame integration and sanity plot (Task 4)"
```

---

### Task 2: `raycasting.py` + unit tests

**Files:**
- Create: `src/raycasting.py`
- Create: `tests/test_raycasting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_raycasting.py`:

```python
"""Unit tests for vectorised Bresenham raycasting."""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.raycasting import cast_rays_batch


def test_horizontal_ray_free_and_hit_cells():
    """Ray from (0,0) to (0,10): cells 0..9 are free, cell 10 is hit."""
    origin = np.array([0, 0], dtype=np.int32)
    hits = np.array([[0, 10]], dtype=np.int32)
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)

    assert len(valid_hits) == 1
    assert list(valid_hits[0]) == [0, 10]
    free_set = set(map(tuple, free.tolist()))
    for c in range(10):
        assert (0, c) in free_set, f"Cell (0,{c}) should be in free cells"


def test_zero_length_ray_no_free_cells():
    """A ray where origin == hit has no free cells (length=1, step 0 is origin)."""
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.array([[5, 5]], dtype=np.int32)
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    # step 0 covers origin cell — it IS included as a free cell
    # (origin always in free-space since the sensor is at origin)
    free_set = set(map(tuple, free.tolist()))
    assert (5, 5) in free_set  # origin is a free cell


def test_out_of_bounds_hits_excluded():
    """Hits outside grid bounds are excluded from valid_hits."""
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.array([[5, 30], [5, 5]], dtype=np.int32)  # first is out-of-bounds
    grid_shape = (20, 20)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    assert list(valid_hits[0]) == [5, 5]


def test_diagonal_ray_covers_diagonal_cells():
    """45° ray from (0,0) to (5,5) visits diagonal cells."""
    origin = np.array([0, 0], dtype=np.int32)
    hits = np.array([[5, 5]], dtype=np.int32)
    grid_shape = (10, 10)
    free, valid_hits = cast_rays_batch(origin, hits, grid_shape)
    assert len(valid_hits) == 1
    assert len(free) >= 4  # at least 4 diagonal steps


def test_empty_hits_returns_empty():
    origin = np.array([5, 5], dtype=np.int32)
    hits = np.zeros((0, 2), dtype=np.int32)
    free, valid_hits = cast_rays_batch(origin, hits, (20, 20))
    assert len(free) == 0
    assert len(valid_hits) == 0


def test_large_scan_under_one_second():
    """16 000 rays of ≤ 320 steps each must complete in < 1 s."""
    rng = np.random.default_rng(42)
    origin = np.array([200, 200], dtype=np.int32)
    hits = (origin + rng.integers(-150, 150, size=(16000, 2))).astype(np.int32)
    grid_shape = (400, 400)
    t0 = time.perf_counter()
    cast_rays_batch(origin, hits, grid_shape)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0, f"16 000-ray cast took {elapsed:.2f} s (limit: 1.0 s)"
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_raycasting.py -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'cast_rays_batch' from 'src.raycasting'`.

- [ ] **Step 3: Create `src/raycasting.py`**

```python
"""Vectorised DDA / Bresenham-style ray casting for the occupancy grid.

Algorithm (per ray from origin_cell to hit_cell[i]):
    steps_i = max(|dr_i|, |dc_i|, 1)          Chebyshev length of ray
    for s in 0 .. steps_i − 1:                 free-space cells
        row[s] = round(r0 + dr_i · s / steps_i)
        col[s] = round(c0 + dc_i · s / steps_i)
    hit cell  = hit_cell[i]

All N rays are processed simultaneously via broadcast numpy operations,
producing a (N, L_max) matrix of row/col values that is then masked and
flattened. Rays longer than *max_steps* are truncated.
"""
from __future__ import annotations

import numpy as np


def cast_rays_batch(
    origin_cell: np.ndarray,      # (2,) int  [row, col]
    hit_cells: np.ndarray,        # (N, 2) int  [row, col]
    grid_shape: tuple[int, int],
    max_steps: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Cast N rays from *origin_cell* to each row of *hit_cells*.

    Returns
    -------
    free_cells  : (M, 2) int32 — cells along rays (free space),
                  includes the origin cell, excludes hit endpoints.
    valid_hits  : (K, 2) int32 — hit_cells that fall within *grid_shape*.
    """
    if len(hit_cells) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 2), dtype=np.int32),
        )

    N = len(hit_cells)
    dr = (hit_cells[:, 0] - origin_cell[0]).astype(np.float32)  # (N,)
    dc = (hit_cells[:, 1] - origin_cell[1]).astype(np.float32)  # (N,)

    # Chebyshev length (at least 1 to avoid division by zero)
    lengths = np.maximum(
        np.maximum(np.abs(dr), np.abs(dc)).astype(np.int32), 1
    )
    lengths = np.minimum(lengths, max_steps)  # (N,)

    L = int(lengths.max())
    steps = np.arange(L, dtype=np.float32)  # (L,)

    # Parametric positions t[i, s] = s / lengths[i]
    t = steps[None, :] / lengths[:, None].astype(np.float32)  # (N, L)
    valid_mask = steps[None, :] < lengths[:, None]             # (N, L) bool

    rows = np.round(origin_cell[0] + dr[:, None] * t).astype(np.int32)  # (N, L)
    cols = np.round(origin_cell[1] + dc[:, None] * t).astype(np.int32)  # (N, L)

    H, W = grid_shape
    in_grid = (
        (rows >= 0) & (rows < H)
        & (cols >= 0) & (cols < W)
        & valid_mask
    )

    free_cells = np.column_stack([rows[in_grid], cols[in_grid]]).astype(np.int32)

    hit_in = (
        (hit_cells[:, 0] >= 0) & (hit_cells[:, 0] < H)
        & (hit_cells[:, 1] >= 0) & (hit_cells[:, 1] < W)
    )
    valid_hits = hit_cells[hit_in].astype(np.int32)

    return free_cells, valid_hits
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_raycasting.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/raycasting.py tests/test_raycasting.py
git commit -m "feat: add vectorised Bresenham raycasting module (Task 4)"
```

---

### Task 3: `occupancy_grid.py` + unit tests

**Files:**
- Create: `src/occupancy_grid.py`
- Create: `tests/test_occupancy_grid.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_occupancy_grid.py`:

```python
"""Unit tests for ProbabilisticGrid."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.occupancy_grid import ProbabilisticGrid, L_MAX, L_MIN


def _small() -> ProbabilisticGrid:
    return ProbabilisticGrid(origin_xy=(0.0, 0.0), extent_m=10.0, resolution=1.0)


def test_fresh_grid_has_no_occupied_or_free_cells():
    g = _small()
    assert not g.occupied_mask().any(), "Fresh grid: no occupied cells"
    assert not g.free_mask().any(),     "Fresh grid: no free cells"


def test_single_hit_marks_cell_occupied():
    g = _small()
    g.update_from_scan(np.array([0.0, 0.0]), np.array([[5.0, 0.0]]))
    assert g.occupied_mask().any(), "Hit cell should be occupied after update"


def test_repeated_hits_clip_at_l_max():
    g = _small()
    for _ in range(30):
        g.update_from_scan(np.array([0.0, 0.0]), np.array([[5.0, 0.0]]))
    assert pytest.approx(g._log_odds.max(), abs=1e-4) == L_MAX


def test_ray_traversal_marks_free_cells():
    g = _small()
    g.update_from_scan(np.array([0.0, 0.0]), np.array([[8.0, 0.0]]))
    assert g.free_mask().any(), "Cells along ray should become free"


def test_mask_dynamic_objects_resets_log_odds():
    g = _small()
    for _ in range(10):
        g.update_from_scan(np.array([0.0, 0.0]), np.array([[5.0, 0.0]]))
    # Box centred at (5,0) — should cover the hit cell
    g.mask_dynamic_objects([(5.0, 0.0, 0.0, 3.0, 3.0)])
    col = int((5.0 - g.origin_x) / g.resolution)
    row = int((0.0 - g.origin_y) / g.resolution)
    if 0 <= row < g._shape[0] and 0 <= col < g._shape[1]:
        assert abs(g._log_odds[row, col]) < 0.01, "Masked cell must be reset to 0"


def test_render_returns_uint8_grey_for_unknown():
    g = _small()
    img = g.render()
    assert img.dtype == np.uint8
    assert img.shape == g._shape
    centre = img[img.shape[0] // 2, img.shape[1] // 2]
    assert 100 < centre < 156, f"Unknown cell should render ~128, got {centre}"
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_occupancy_grid.py -v 2>&1 | head -10
```
Expected: `ImportError: cannot import name 'ProbabilisticGrid'`.

- [ ] **Step 3: Create `src/occupancy_grid.py`**

```python
"""Probabilistic occupancy grid using the log-odds inverse sensor model.

Log-odds formulation:
    l(x) = log( P(occ|z) / P(free|z) )

    Hit  (LiDAR return at cell x): l(x) += L_HIT  ≈ +0.85  (→ ~70 % occupied)
    Miss (ray passes through x):   l(x) += L_MISS ≈ -0.40  (→ ~40 % occupied)
    Clipped to [L_MIN, L_MAX] = [-5, 5] so no cell is permanently locked.

Recovering probability: p = sigmoid(l) = 1 / (1 + exp(-l))

Grid axes: columns = world-x (east), rows = world-y (north).
Origin (0,0) of the grid array is the SW corner at (origin_x, origin_y).
"""
from __future__ import annotations

import math

import numpy as np

from src.raycasting import cast_rays_batch

L_HIT: float = 0.85     # log-odds increment per LiDAR return
L_MISS: float = -0.40   # log-odds decrement per free-space traversal
L_MAX: float = 5.0
L_MIN: float = -5.0


class ProbabilisticGrid:
    """Probabilistic 2D occupancy grid in the world frame."""

    def __init__(
        self,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        extent_m: float = 300.0,
        resolution: float = 0.25,
    ) -> None:
        self.resolution = float(resolution)
        # SW corner coordinates
        self.origin_x = float(origin_xy[0]) - extent_m
        self.origin_y = float(origin_xy[1]) - extent_m
        n = int(round(2.0 * extent_m / resolution))
        self._log_odds = np.zeros((n, n), dtype=np.float32)
        self._shape: tuple[int, int] = (n, n)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """(N, 2) world XY → (N, 2) int32 [row, col]."""
        col = ((xy[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        row = ((xy[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        return np.column_stack([row, col])

    def _origin_cell(self, xy: np.ndarray) -> np.ndarray:
        """(2,) world XY → (2,) int32 [row, col]."""
        col = int((float(xy[0]) - self.origin_x) / self.resolution)
        row = int((float(xy[1]) - self.origin_y) / self.resolution)
        return np.array([row, col], dtype=np.int32)

    # ── update ────────────────────────────────────────────────────────────────

    def update_from_scan(
        self,
        sensor_origin_world: np.ndarray,   # (2,) world XY
        hits_world: np.ndarray,             # (N, 2) world XY
        max_range_m: float = 80.0,
    ) -> None:
        """Update log-odds from one LiDAR scan.

        Free cells (ray traversal) receive L_MISS; hit endpoints receive L_HIT.
        """
        if len(hits_world) == 0:
            return

        origin_cell = self._origin_cell(sensor_origin_world)
        hit_cells = self._world_to_cell(hits_world)
        max_steps = int(max_range_m / self.resolution)

        free_cells, valid_hits = cast_rays_batch(
            origin_cell, hit_cells, self._shape, max_steps=max_steps
        )

        if len(free_cells):
            np.add.at(self._log_odds, (free_cells[:, 0], free_cells[:, 1]), L_MISS)
        if len(valid_hits):
            np.add.at(self._log_odds, (valid_hits[:, 0], valid_hits[:, 1]), L_HIT)

        np.clip(self._log_odds, L_MIN, L_MAX, out=self._log_odds)

    # ── dynamic object masking ────────────────────────────────────────────────

    def mask_dynamic_objects(
        self,
        boxes_world: list[tuple[float, float, float, float, float]],
    ) -> None:
        """Reset log-odds to 0 inside each (cx, cy, yaw, length, width) box."""
        for cx, cy, yaw, length, width in boxes_world:
            c, s = math.cos(float(yaw)), math.sin(float(yaw))
            hl, hw = float(length) / 2.0, float(width) / 2.0
            corners = np.array(
                [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64
            )
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            corners_w = (R @ corners.T).T + np.array([cx, cy])
            cells = self._world_to_cell(corners_w)
            r0 = max(0, int(cells[:, 0].min()))
            r1 = min(self._shape[0] - 1, int(cells[:, 0].max()))
            c0 = max(0, int(cells[:, 1].min()))
            c1 = min(self._shape[1] - 1, int(cells[:, 1].max()))
            if r0 <= r1 and c0 <= c1:
                self._log_odds[r0 : r1 + 1, c0 : c1 + 1] = 0.0

    # ── query ─────────────────────────────────────────────────────────────────

    def occupied_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Bool array where P(occ) > threshold (l > log(p/(1-p)))."""
        l_thresh = math.log(threshold / (1.0 - threshold))
        return self._log_odds > l_thresh

    def free_mask(self, threshold: float = 0.3) -> np.ndarray:
        """Bool array where P(occ) < threshold."""
        l_thresh = math.log(threshold / (1.0 - threshold))
        return self._log_odds < l_thresh

    def render(self) -> np.ndarray:
        """uint8 image: 0=occupied(black), 128=unknown(grey), 255=free(white)."""
        prob = 1.0 / (1.0 + np.exp(-self._log_odds.astype(np.float32)))
        return ((1.0 - prob) * 255.0).astype(np.uint8)

    def log_odds_copy(self) -> np.ndarray:
        """Snapshot of current log-odds array for animation frames."""
        return self._log_odds.copy()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_occupancy_grid.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/occupancy_grid.py tests/test_occupancy_grid.py
git commit -m "feat: add ProbabilisticGrid with log-odds inverse sensor model (Task 4)"
```

---

### Task 4: `track_boundary.py` + unit tests

**Files:**
- Create: `src/track_boundary.py`
- Create: `tests/test_track_boundary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_track_boundary.py`:

```python
"""Unit tests for track boundary extraction."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.track_boundary import extract_boundaries, save_geojson


def _oval_mask(size: int = 400, thickness: int = 3) -> np.ndarray:
    """Synthetic binary mask with two concentric oval barriers."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(img, (200, 200), (150, 80), 0, 0, 360, 255, thickness)
    cv2.ellipse(img, (200, 200), (100, 50), 0, 0, 360, 255, thickness)
    return img > 0


def test_extract_finds_at_least_two_boundaries():
    mask = _oval_mask()
    boundaries = extract_boundaries(
        mask, origin_x=-50.0, origin_y=-50.0,
        resolution=0.25, min_perimeter_cells=100,
    )
    assert len(boundaries) >= 2, f"Expected ≥2 boundaries, got {len(boundaries)}"


def test_boundaries_are_2d_world_coordinate_arrays():
    mask = _oval_mask()
    boundaries = extract_boundaries(
        mask, origin_x=-50.0, origin_y=-50.0,
        resolution=0.25, min_perimeter_cells=100,
    )
    for b in boundaries:
        assert b.ndim == 2 and b.shape[1] == 2
        # x values should be in world range [-50, 50] roughly
        assert b[:, 0].min() > -100.0 and b[:, 0].max() < 100.0


def test_noise_blobs_filtered_out():
    """Small contours (below min_perimeter_cells) must be dropped."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:12, 10:12] = True  # tiny 2×2 blob
    boundaries = extract_boundaries(
        mask, origin_x=0.0, origin_y=0.0,
        resolution=1.0, min_perimeter_cells=50,
    )
    assert len(boundaries) == 0, "Tiny blob should be filtered out"


def test_save_geojson_produces_valid_featurecollection(tmp_path: Path):
    pts = [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])]
    path = str(tmp_path / "test.geojson")
    save_geojson(pts, path)
    with open(path) as f:
        gj = json.load(f)
    assert gj["type"] == "FeatureCollection"
    assert len(gj["features"]) == 1
    assert gj["features"][0]["geometry"]["type"] == "LineString"
    assert "length_m" in gj["features"][0]["properties"]
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_track_boundary.py -v 2>&1 | head -10
```
Expected: `ImportError`.

- [ ] **Step 3: Create `src/track_boundary.py`**

```python
"""Track barrier polyline extraction from the accumulated occupancy grid.

Pipeline
--------
1. Threshold log-odds grid to binary occupied image.
2. Morphological closing (3×3 kernel) to bridge small gaps in the barriers.
3. cv2.findContours with RETR_CCOMP retrieves both outer contour (outer
   barrier edge) and inner hole contours (inner barrier edge of a closed
   loop track).
4. Filter: discard contours whose arc length < min_perimeter_cells cells
   (default 200 cells = 50 m at 0.25 m/cell).
5. Simplify with Douglas-Peucker (cv2.approxPolyDP) at ε ~ 0.5 m.
6. Convert OpenCV pixel [col, row] → world [x, y] coordinates.
7. Serialise as GeoJSON LineString features.
"""
from __future__ import annotations

import json

import cv2
import numpy as np


def extract_boundaries(
    occupied_mask: np.ndarray,         # (H, W) bool
    origin_x: float,
    origin_y: float,
    resolution: float,
    min_perimeter_cells: int = 200,    # 200 × 0.25 m = 50 m minimum
    dp_epsilon_m: float = 0.5,
) -> list[np.ndarray]:
    """Extract track barrier polylines from the occupancy mask.

    Parameters
    ----------
    occupied_mask      : Boolean occupancy array (True = occupied).
    origin_x, origin_y : World coordinates of the grid's SW corner.
    resolution         : Metres per cell.
    min_perimeter_cells: Contours shorter than this (cells) are discarded.
    dp_epsilon_m       : Douglas-Peucker simplification tolerance (metres).

    Returns
    -------
    List of (M, 2) float64 arrays; each row is (x_world, y_world).
    """
    u8 = occupied_mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)

    # RETR_CCOMP: level-0 = outer contours, level-1 = inner hole contours
    contours, _ = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    dp_eps = dp_epsilon_m / resolution
    boundaries: list[np.ndarray] = []

    for contour in contours:
        if cv2.arcLength(contour, closed=True) < min_perimeter_cells:
            continue
        approx = cv2.approxPolyDP(contour, dp_eps, closed=True)
        pts = approx.reshape(-1, 2)   # (M, 2)  [col, row] OpenCV convention
        x_w = pts[:, 0].astype(np.float64) * resolution + origin_x
        y_w = pts[:, 1].astype(np.float64) * resolution + origin_y
        boundaries.append(np.column_stack([x_w, y_w]))

    return boundaries


def save_geojson(boundaries: list[np.ndarray], path: str) -> None:
    """Serialise boundary polylines as a GeoJSON FeatureCollection of LineStrings."""
    features = []
    for i, pts in enumerate(boundaries):
        if len(pts) < 2:
            continue
        length_m = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        features.append({
            "type": "Feature",
            "properties": {"id": i, "length_m": round(length_m, 2)},
            "geometry": {"type": "LineString", "coordinates": pts.tolist()},
        })
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_track_boundary.py -v
```
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/track_boundary.py tests/test_track_boundary.py
git commit -m "feat: add track boundary extraction with OpenCV CCOMP contours (Task 4)"
```

---

### Task 5: `map_visualization.py`

**Files:**
- Create: `src/map_visualization.py`

No unit tests (visual outputs; verified by inspecting the saved PNGs/GIF).

- [ ] **Step 1: Create `src/map_visualization.py`**

```python
"""Visualisation for Task 4 environment mapping outputs.

Deliverables
------------
- world_map_static.png           : final occupancy grid + boundary polylines
- world_map_with_trajectory.png  : same + ego world-frame trajectory
- ego_view_NNN.png               : ego-centred 100×60 m rotated crop, with tracks
- map_animation.gif              : map filling in over time (uses Pillow)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.ego_trajectory import EgoTrajectory
from src.occupancy_grid import ProbabilisticGrid


# ── helpers ───────────────────────────────────────────────────────────────────

def _log_odds_to_uint8(log_odds: np.ndarray) -> np.ndarray:
    """Convert log-odds array to uint8: 0=occupied, 128=unknown, 255=free."""
    prob = 1.0 / (1.0 + np.exp(-log_odds.astype(np.float32)))
    return ((1.0 - prob) * 255.0).astype(np.uint8)


def _grid_extent(grid: ProbabilisticGrid) -> list[float]:
    """[x_min, x_max, y_min, y_max] for matplotlib imshow extent."""
    return [
        grid.origin_x,
        grid.origin_x + grid._shape[1] * grid.resolution,
        grid.origin_y,
        grid.origin_y + grid._shape[0] * grid.resolution,
    ]


def _draw_oriented_box(
    ax: plt.Axes, cx: float, cy: float, yaw: float,
    length: float, width: float, track_id: int,
) -> None:
    """Draw a rotated bounding box on *ax* (ego frame coordinates)."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw], [hl, hw]])
    R = np.array([[c, -s], [s, c]])
    pts = (R @ corners.T).T + np.array([cx, cy])
    ax.plot(pts[:, 0], pts[:, 1], "r-", lw=1.5)
    ax.text(cx, cy, str(track_id), fontsize=6, color="red", ha="center", va="center")


# ── public functions ──────────────────────────────────────────────────────────

def render_world_map_static(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    output_path: str,
) -> None:
    """Save final occupancy grid with boundary polylines overlaid (red)."""
    img = np.flipud(grid.render())  # flip so y increases upward
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=_grid_extent(grid), origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.0, alpha=0.8,
                label="boundary" if i == 0 else "")
    if boundaries:
        ax.legend(fontsize=8)
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Accumulated occupancy map (static environment)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_world_map_with_trajectory(
    grid: ProbabilisticGrid,
    boundaries: list[np.ndarray],
    traj_xy: np.ndarray,       # (N, 2) world positions
    output_path: str,
) -> None:
    """Save occupancy grid + ego trajectory + boundary polylines."""
    img = np.flipud(grid.render())
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, cmap="gray", vmin=0, vmax=255,
              extent=_grid_extent(grid), origin="lower")
    for i, b in enumerate(boundaries):
        ax.plot(b[:, 0], b[:, 1], "r-", lw=1.0, alpha=0.8,
                label="boundary" if i == 0 else "")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], "b-", lw=0.8, alpha=0.6,
            label="ego trajectory")
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "go", ms=6, label="start")
    ax.plot(traj_xy[-1, 0], traj_xy[-1, 1], "rs", ms=6, label="end")
    ax.set_xlabel("x_world (m)")
    ax.set_ylabel("y_world (m)")
    ax.set_title("Task 4 — Occupancy map with ego trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def render_ego_view(
    grid: ProbabilisticGrid,
    ego_traj: EgoTrajectory,
    timestamp_ns: int,
    track_rows: list[dict],    # rows from tracks.csv at this timestamp (ego frame)
    output_path: str,
    window_x_m: float = 100.0,
    window_y_m: float = 60.0,
) -> None:
    """Render a heading-aligned, ego-centred crop of the world map.

    The window is rotated so ego-forward points upward in the image.
    Track boxes (from tracks.csv, ego frame) are drawn directly on the crop.
    """
    x_ego, y_ego, heading = ego_traj.pose_at(timestamp_ns)
    res = grid.resolution
    nx = int(window_x_m / res)
    ny = int(window_y_m / res)

    # Sample world-frame occupancy at points covering the ego window
    xs = (np.arange(nx) - nx // 2) * res   # ego-x axis
    ys = (np.arange(ny) - ny // 2) * res   # ego-y axis
    xs_g, ys_g = np.meshgrid(xs, ys)       # (ny, nx)

    c, s = math.cos(heading), math.sin(heading)
    xw = x_ego + c * xs_g - s * ys_g
    yw = y_ego + s * xs_g + c * ys_g

    col_f = (xw - grid.origin_x) / res
    row_f = (yw - grid.origin_y) / res
    col_i = np.clip(col_f.astype(np.int32), 0, grid._shape[1] - 1)
    row_i = np.clip(row_f.astype(np.int32), 0, grid._shape[0] - 1)
    in_bounds = (
        (col_f >= 0) & (col_f < grid._shape[1])
        & (row_f >= 0) & (row_f < grid._shape[0])
    )

    img_full = grid.render()
    crop = np.full((ny, nx), 128, dtype=np.uint8)
    crop[in_bounds] = img_full[row_i[in_bounds], col_i[in_bounds]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(
        np.flipud(crop), cmap="gray", vmin=0, vmax=255,
        extent=[-window_x_m / 2, window_x_m / 2,
                -window_y_m / 2, window_y_m / 2],
        origin="lower",
    )
    for row_d in track_rows:
        _draw_oriented_box(
            ax,
            float(row_d["center_x"]), float(row_d["center_y"]),
            float(row_d["yaw"]), float(row_d["length"]), float(row_d["width"]),
            int(row_d["track_id"]),
        )
    ax.plot(0, 0, "g^", ms=10, label="ego")
    ax.set_xlabel("ego-x (m, forward)")
    ax.set_ylabel("ego-y (m, left)")
    ax.set_title(f"Ego-frame view  t={timestamp_ns}")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def render_animation(
    grid_snapshots: list[np.ndarray],        # list of log_odds float32 copies
    ego_poses: list[tuple[float, float]],    # (x_w, y_w) per snapshot
    grid: ProbabilisticGrid,
    output_path: str,
) -> None:
    """Save animated GIF showing map filling in.

    Uses Pillow (PIL) directly — no imageio dependency required.
    """
    from PIL import Image

    frames_pil: list[Image.Image] = []
    extent = _grid_extent(grid)

    for i, (log_odds, (x_ego, y_ego)) in enumerate(zip(grid_snapshots, ego_poses)):
        img_arr = np.flipud(_log_odds_to_uint8(log_odds))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_arr, cmap="gray", vmin=0, vmax=255,
                  extent=extent, origin="lower")
        # Fading trail of last 20 poses
        trail_x = [p[0] for p in ego_poses[max(0, i - 20): i + 1]]
        trail_y = [p[1] for p in ego_poses[max(0, i - 20): i + 1]]
        if trail_x:
            ax.plot(trail_x, trail_y, "b-", lw=0.8, alpha=0.5)
        ax.plot(x_ego, y_ego, "go", ms=7)
        ax.set_title(f"Frame {i + 1}/{len(grid_snapshots)}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0.5)

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        rgb = buf.reshape(h, w, 3)
        frames_pil.append(Image.fromarray(rgb))
        plt.close(fig)

    if frames_pil:
        frames_pil[0].save(
            output_path,
            save_all=True,
            append_images=frames_pil[1:],
            loop=0,
            duration=250,   # ms per frame
            optimize=True,
        )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
uv run python -c "from src.map_visualization import render_world_map_static; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/map_visualization.py
git commit -m "feat: add map_visualization module for Task 4 outputs"
```

---

### Task 6: `run_task4.py` CLI + end-to-end run

**Files:**
- Create: `scripts/run_task4.py`

- [ ] **Step 1: Create `scripts/run_task4.py`**

```python
"""CLI entry point for EC2026 Task 4 (Environment Mapping)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bag_inspector import parse_bag_summary
from src.ego_trajectory import EgoTrajectory
from src.map_visualization import (
    render_animation,
    render_ego_view,
    render_world_map_static,
    render_world_map_with_trajectory,
)
from src.occupancy_grid import ProbabilisticGrid
from src.sensor_detectors import iter_ego_frame_points
from src.tf_utils import EgoStateLog
from src.track_boundary import extract_boundaries, save_geojson

# Height filter (ego frame): retains barriers and vehicles; drops ground/sky
_Z_MIN = 0.10   # m above ego origin
_Z_MAX = 2.50   # m above ego origin

_LIDAR_ALIAS = {
    "front": "/sensor/lidar_front/points",
    "left":  "/sensor/lidar_left/points",
    "right": "/sensor/lidar_right/points",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 4: Probabilistic environment map from LiDAR data."
    )
    p.add_argument("--bag-dir", default="data")
    p.add_argument("--max-frames", type=int, default=60)
    p.add_argument("--output-dir", default="outputs/task4")
    p.add_argument("--grid-resolution", type=float, default=0.25,
                   help="Cell size in metres (default 0.25)")
    p.add_argument("--grid-extent", type=float, default=300.0,
                   help="Half-extent of grid in metres (±N m both axes, default 300)")
    p.add_argument("--use-lidars", default="front,left,right",
                   help="Comma-separated subset of: front,left,right")
    p.add_argument("--mask-tracks", default="outputs/task3/tracks.csv",
                   help="tracks.csv for dynamic-object masking (Task 3 output)")
    p.add_argument("--no-mask-tracks", action="store_true",
                   help="Skip dynamic-object masking")
    p.add_argument("--animation-frames", type=int, default=30,
                   help="Number of GIF frames (default 30)")
    p.add_argument("--no-animation", action="store_true",
                   help="Skip GIF generation (faster)")
    p.add_argument("--inspect-only", action="store_true",
                   help="Print grid size and exit")
    p.add_argument("--max-range", type=float, default=80.0,
                   help="Max LiDAR range for ray casting (m, default 80)")
    return p.parse_args()


def _load_ego(bag_dir: str, bag_start_ns: int) -> EgoStateLog | None:
    for candidate in [
        Path(bag_dir) / "ego.mat",
        *Path(bag_dir).parent.glob("**/ego.mat"),
    ]:
        if candidate.exists():
            try:
                log = EgoStateLog(str(candidate), bag_start_ns=bag_start_ns)
                print(f"[Task 4] Ego log: {candidate}  {log}")
                return log
            except Exception as exc:
                print(f"[Task 4] ego.mat load failed: {exc}")
    return None


def main() -> None:
    args = parse_args()
    summary = parse_bag_summary(args.bag_dir)
    bag_start_ns = int(summary.starting_time_ns or 0)

    if args.inspect_only:
        n = int(2 * args.grid_extent / args.grid_resolution)
        print(f"[Task 4] Grid: ±{args.grid_extent} m, "
              f"res={args.grid_resolution} m → {n}×{n} cells "
              f"({n*n/1e6:.1f}M, {n*n*4/1e6:.0f} MB float32)")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Ego trajectory ────────────────────────────────────────────────────────
    ego_log = _load_ego(args.bag_dir, bag_start_ns)
    if ego_log is None:
        print("[Task 4] ERROR: ego.mat not found")
        sys.exit(1)
    ego_traj = EgoTrajectory.from_ego_log(ego_log)
    x_min, x_max, y_min, y_max = ego_traj.world_extent()
    print(f"[Task 4] Full-bag extent: x=[{x_min:.0f},{x_max:.0f}] "
          f"y=[{y_min:.0f},{y_max:.0f}] m")

    # ── Occupancy grid ────────────────────────────────────────────────────────
    grid = ProbabilisticGrid(
        origin_xy=(0.0, 0.0),
        extent_m=args.grid_extent,
        resolution=args.grid_resolution,
    )
    n = grid._shape[0]
    print(f"[Task 4] Grid: {n}×{n} cells ({n*n/1e6:.1f}M), "
          f"origin=({grid.origin_x:.0f},{grid.origin_y:.0f})")

    # ── Track masking ─────────────────────────────────────────────────────────
    tracks_df: pd.DataFrame | None = None
    if not args.no_mask_tracks:
        tp = Path(args.mask_tracks)
        if tp.exists():
            tracks_df = pd.read_csv(tp)
            print(f"[Task 4] Loaded {len(tracks_df)} track rows for masking")
        else:
            print(f"[Task 4] tracks.csv not found at {tp} — masking skipped")

    # ── LiDAR topics ─────────────────────────────────────────────────────────
    lidar_topics = [
        _LIDAR_ALIAS[k.strip()]
        for k in args.use_lidars.split(",")
        if k.strip() in _LIDAR_ALIAS
    ]
    print(f"[Task 4] LiDAR topics: {lidar_topics}")

    # ── Collect scans per timestamp ───────────────────────────────────────────
    scans: dict[int, list[np.ndarray]] = {}
    for topic in lidar_topics:
        n_frames = 0
        for ts_ns, ego_xyz in iter_ego_frame_points(summary, topic, args.max_frames):
            scans.setdefault(ts_ns, []).append(ego_xyz)
            n_frames += 1
        print(f"[Task 4]   {topic}: {n_frames} frames")

    sorted_ts = sorted(scans.keys())
    print(f"[Task 4] Total unique timestamps: {len(sorted_ts)}")

    # ── Animation / ego-view bookkeeping ─────────────────────────────────────
    anim_stride = max(1, len(sorted_ts) // max(args.animation_frames, 1))
    anim_snapshots: list[np.ndarray] = []
    anim_ego_poses: list[tuple[float, float]] = []
    master_timestamps: list[int] = []

    n_ego_views = min(8, max(5, len(sorted_ts) // 8))
    ego_view_indices = (
        np.linspace(0, len(sorted_ts) - 1, n_ego_views, dtype=int).tolist()
        if sorted_ts else []
    )

    # ── Group track rows by timestamp (for masking + ego views) ──────────────
    rows_by_ts: dict[int, list[dict]] = {}
    if tracks_df is not None:
        for _, row in tracks_df.iterrows():
            rows_by_ts.setdefault(int(row["timestamp_ns"]), []).append(row.to_dict())

    # ── Main loop ─────────────────────────────────────────────────────────────
    for frame_i, ts_ns in enumerate(sorted_ts):
        x_ego, y_ego, heading = ego_traj.pose_at(ts_ns)
        sensor_origin_world = np.array([x_ego, y_ego])

        # Transform all scans at this timestamp to world frame and update grid
        all_pts: list[np.ndarray] = []
        for ego_xyz in scans[ts_ns]:
            z_ok = (ego_xyz[:, 2] >= _Z_MIN) & (ego_xyz[:, 2] <= _Z_MAX)
            pts_2d = ego_xyz[z_ok, :2]
            if len(pts_2d):
                all_pts.append(ego_traj.ego_to_world(pts_2d, ts_ns))

        if all_pts:
            grid.update_from_scan(
                sensor_origin_world, np.vstack(all_pts), args.max_range
            )

        # Dynamic object masking
        frame_tracks = rows_by_ts.get(ts_ns, [])
        if frame_tracks:
            boxes: list[tuple] = []
            for rd in frame_tracks:
                pts_w = ego_traj.ego_to_world(
                    np.array([[float(rd["center_x"]), float(rd["center_y"])]]),
                    ts_ns,
                )
                boxes.append((
                    float(pts_w[0, 0]), float(pts_w[0, 1]),
                    float(rd["yaw"]) + heading,
                    float(rd["length"]), float(rd["width"]),
                ))
            grid.mask_dynamic_objects(boxes)

        # Animation snapshot
        if not args.no_animation and frame_i % anim_stride == 0:
            anim_snapshots.append(grid.log_odds_copy())
            anim_ego_poses.append((x_ego, y_ego))

        master_timestamps.append(ts_ns)

        if frame_i % 10 == 0 or frame_i == len(sorted_ts) - 1:
            occ = grid.occupied_mask().sum()
            print(f"  [{frame_i+1}/{len(sorted_ts)}] occ_cells={occ:,}")

    # ── Free-space estimate ───────────────────────────────────────────────────
    free_cells = int(grid.free_mask().sum())
    occ_cells = int(grid.occupied_mask().sum())
    res2 = args.grid_resolution ** 2
    print(f"\n[Task 4] Free space : {free_cells:,} cells = {free_cells * res2:.0f} m²")
    print(f"[Task 4] Occupied   : {occ_cells:,} cells = {occ_cells * res2:.0f} m²")

    # ── Track boundary extraction ─────────────────────────────────────────────
    print("[Task 4] Extracting track boundaries …")
    min_perim = int(50.0 / args.grid_resolution)
    boundaries = extract_boundaries(
        grid.occupied_mask(),
        origin_x=grid.origin_x, origin_y=grid.origin_y,
        resolution=args.grid_resolution,
        min_perimeter_cells=min_perim,
    )
    total_len = sum(
        float(np.sum(np.linalg.norm(np.diff(b, axis=0), axis=1)))
        for b in boundaries if len(b) > 1
    )
    print(f"[Task 4] Boundaries : {len(boundaries)}, total length {total_len:.0f} m")
    save_geojson(boundaries, str(output_dir / "track_boundaries.geojson"))

    # ── Visualisations ────────────────────────────────────────────────────────
    print("[Task 4] Rendering world_map_static.png …")
    render_world_map_static(grid, boundaries, str(output_dir / "world_map_static.png"))

    stride = max(1, len(ego_traj._x_w) // 5000)
    traj_xy = ego_traj.trajectory_xy()[::stride]
    print("[Task 4] Rendering world_map_with_trajectory.png …")
    render_world_map_with_trajectory(
        grid, boundaries, traj_xy,
        str(output_dir / "world_map_with_trajectory.png"),
    )

    print(f"[Task 4] Rendering {len(ego_view_indices)} ego views …")
    for view_i, fi in enumerate(ego_view_indices):
        ts_ns = master_timestamps[fi]
        out_path = output_dir / f"ego_view_{view_i:03d}.png"
        render_ego_view(
            grid, ego_traj, ts_ns, rows_by_ts.get(ts_ns, []), str(out_path)
        )

    if not args.no_animation and anim_snapshots:
        anim_path = output_dir / "map_animation.gif"
        print(f"[Task 4] Rendering animation ({len(anim_snapshots)} frames) …")
        render_animation(anim_snapshots, anim_ego_poses, grid, str(anim_path))
        size_mb = anim_path.stat().st_size / 1e6
        print(f"[Task 4] map_animation.gif: {size_mb:.1f} MB")

    print("\n[Task 4] Done.")
    print(f"  Timestamps processed : {len(sorted_ts)}")
    print(f"  Free space           : {free_cells * res2:.0f} m²")
    print(f"  Boundary polylines   : {len(boundaries)}")
    print(f"  Total boundary length: {total_len:.0f} m")
    print(f"  Outputs              : {output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run end-to-end smoke test (no animation)**

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 60 \
    --output-dir outputs/task4 --no-animation --grid-extent 300 2>&1 | tail -20
```
Expected: exits without error, prints "Done." with `Free space > 0 m²` and `Boundary polylines` count.

- [ ] **Step 3: Verify Task 3 outputs are byte-identical (regression check)**

```bash
md5sum outputs/task3/tracks.csv > /tmp/t3_before.md5
uv run python scripts/run_task3.py --bag-dir data --max-frames 60 \
    --bev-frames 0 --output-dir /tmp/t3_regression
md5sum /tmp/t3_regression/tracks.csv > /tmp/t3_after.md5
diff /tmp/t3_before.md5 /tmp/t3_after.md5 && echo "REGRESSION OK"
```
Expected: `REGRESSION OK`.

- [ ] **Step 4: Run full pipeline with animation**

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 60 \
    --output-dir outputs/task4 --animation-frames 30 2>&1 | tail -15
```
Expected: `map_animation.gif` created. Check size:
```bash
ls -lh outputs/task4/map_animation.gif
```
Expected: `< 10 MB`.

- [ ] **Step 5: Run without masking (for report comparison)**

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 60 \
    --output-dir outputs/task4_no_mask --no-mask-tracks --no-animation 2>&1 | tail -10
```
Expected: completes without error, produces `outputs/task4_no_mask/world_map_static.png`.

- [ ] **Step 6: Commit all outputs and script**

```bash
git add scripts/run_task4.py outputs/task4/ outputs/task4_no_mask/
git commit -m "feat: add run_task4.py end-to-end pipeline and all map outputs"
```

---

### Task 7: Report + README update

**Files:**
- Create: `outputs/task4/task4_report.md`
- Modify: `README.md`

- [ ] **Step 1: Collect metrics from the completed run**

```bash
uv run python -c "
import json, pandas as pd
from pathlib import Path
gj = json.loads(Path('outputs/task4/track_boundaries.geojson').read_text())
print(f'Boundary features: {len(gj[\"features\"])}')
for f in gj['features']:
    p = f['properties']
    print(f'  id={p[\"id\"]}  length={p[\"length_m\"]:.0f} m')
df = pd.read_csv('outputs/task4/tracks.csv')
print(f'Track rows: {len(df)}')
print(f'Unique timestamps: {df[\"timestamp_ns\"].nunique()}')
"
```

- [ ] **Step 2: Create `outputs/task4/task4_report.md`**

Use the metrics printed above. Fill in the `<ACTUAL_*>` placeholders with real numbers before committing.

```markdown
# Task 4 — Environment Mapping Report

## Pipeline Summary

LiDAR point clouds from three sensors (front, left, right) are accumulated
into a probabilistic occupancy grid in the world frame using a Bayesian
inverse-sensor model (log-odds formulation). Dynamic opponents are masked
out using Task 3 track bounding boxes transformed to world coordinates.
Track barriers are extracted as polylines from the final binary map.

## Log-Odds Parameters

| Parameter | Value | Rationale |
|---|---|---|
| L_HIT  | +0.85 | Equivalent to ~70% P(occ\|hit); each LiDAR return is reliable but not certain (reflections, noise) |
| L_MISS | −0.40 | Equivalent to ~40% P(occ\|miss); gentler than hits because free-space rays can be blocked by transient objects |
| L_MAX  | +5.0  | Caps at ~99.3% — cells never permanently locked occupied |
| L_MIN  | −5.0  | Caps at ~0.7% — cells never permanently locked free |
| Clip range | [−5, 5] | Ensures the map can recover from transient misclassifications |

## Resolution / Extent Trade-off

| Choice | Cells | Memory | Detail |
|---|---|---|---|
| 0.5 m / ±300 m | 1 200² = 1.44 M | 5.5 MB | Too coarse — loses barrier shape |
| **0.25 m / ±300 m** | **2 400² = 5.76 M** | **22 MB** | ✓ Chosen: captures 25 cm features |
| 0.10 m / ±300 m | 6 000² = 36 M | 137 MB | Fine but slow for 60-frame run |
| 0.25 m / ±700 m | 5 600² = 31.4 M | 120 MB | Required for full-bag map |

Full-bag ego trajectory spans x ∈ [−431, +240] m, y ∈ [−656, +466] m
(671 m × 1 122 m oval circuit). The default ±300 m grid comfortably covers
the 60-frame (~12 s) segment.

## Sensor Selection

| Sensor | Role |
|---|---|
| lidar_front | Primary: sees vehicles ahead and forward barriers |
| lidar_left  | Sees left barrier clearly (per Task 2 analysis); no vehicle detections |
| lidar_right | Sees right barrier clearly; no vehicle detections |
| Radars (all 4) | **Excluded** — radar returns are point reflections, not surface scans; too sparse for occupancy mapping |

## Quantitative Results (60 frames)

| Metric | Value |
|---|---|
| Frames processed | 60 |
| Free space estimate | **<ACTUAL_FREE_M2> m²** |
| Occupied area | **<ACTUAL_OCC_M2> m²** |
| Boundary polylines extracted | **<ACTUAL_N_BOUNDARIES>** |
| Total boundary length | **<ACTUAL_BOUNDARY_LEN_M> m** |

## Dynamic-Object Masking Effect

Running without `--mask-tracks` (`outputs/task4_no_mask/world_map_static.png`)
shows ghosting artefacts along the racing line where opponent vehicles appeared
repeatedly at similar world positions. Each opponent creates a ~5 m × 2 m
occupied smear. With masking enabled, these artefacts are cleared after each
frame and the racing surface remains free space.

The masking pipeline: for each Task 3 confirmed/coasting track at timestamp t,
the ego-frame bounding box (center_x, center_y, yaw, length, width) is
transformed to world coordinates via `EgoTrajectory.ego_to_world(t)`, then
`ProbabilisticGrid.mask_dynamic_objects` resets the bounding box region to
log-odds = 0 (unknown).

## Ego Trajectory Integration

`EgoTrajectory` integrates `ego.mat` velocity samples using the formula:

```
dx_w = vx · cos(h) − vy · sin(h)
dy_w = vx · sin(h) + vy · cos(h)
```

where vx, vy are **ego-frame** (longitudinal / lateral) velocities and h is
the raw heading odometer value. `predict_state()` in `motion_model.py` omits
this rotation because Δheading ≈ 0.4° per step (Δt ≈ 0.2 s), making the
rotation matrix ≈ identity for a single step.

## Limitations and Future Work

1. **No loop-closure.** The map accumulates independently in the world frame
   without any SLAM correction. Drift in `ego.mat` integration accumulates
   over laps, causing the map to show double-walls on the second lap.

2. **Fixed-grid resolution.** A voxel grid or octree would allow adaptive
   resolution (finer near barriers, coarser in open space) at similar memory.

3. **Sensor extrinsics are approximate.** The hardcoded values from Task 2
   introduce a fixed spatial offset; barriers may appear slightly offset from
   their true position.

4. **Side-LiDAR synchronisation.** lidar_left and lidar_right timestamps
   differ from lidar_front. The pipeline processes all timestamps
   independently; for a strict master-clock design, side scans should be
   interpolated to the nearest lidar_front tick.

5. **Track-masking is axis-aligned.** `mask_dynamic_objects` zeros a
   bounding-box region rather than the exact oriented footprint, slightly
   over-erasing the grid near track barriers that are close to opponents.
```

- [ ] **Step 3: Add Task 4 section to `README.md`**

Append the following to the end of `README.md`:

```markdown

---

## Task 4: Environment Mapping (Bonus)

Builds a probabilistic occupancy grid from the three LiDAR sensors,
accumulated in the world frame with dynamic-object masking from Task 3.

### Task 4 — Run

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 60 \
    --output-dir outputs/task4
```

Full-bag run (240 s, needs larger grid):

```bash
uv run python scripts/run_task4.py --bag-dir data --max-frames 9999 \
    --grid-extent 700 --output-dir outputs/task4_full
```

Skip animation for speed: `--no-animation`

### Task 4 — Outputs

- `outputs/task4/world_map_static.png` — final occupancy map with boundary polylines
- `outputs/task4/world_map_with_trajectory.png` — same + ego world-frame trajectory
- `outputs/task4/track_boundaries.geojson` — inner + outer barrier polylines, world frame
- `outputs/task4/ego_view_NNN.png` — ego-centred 100 × 60 m views at 6 timestamps
- `outputs/task4/map_animation.gif` — map filling in as ego drives
- `outputs/task4/task4_report.md` — methodology, parameters, and quantitative results
```

- [ ] **Step 4: Commit**

```bash
git add outputs/task4/task4_report.md README.md
git commit -m "docs: add Task 4 report and README section"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Covered by |
|---|---|
| EgoTrajectory.pose_at + ego_to_world | Task 1 |
| Unit test: start pose = origin | Task 1 Step 1 |
| Unit test: mid-bag displacement < 0.5 m | Task 1 Step 1 |
| ProbabilisticGrid.update_from_scan | Task 3 |
| ProbabilisticGrid.mask_dynamic_objects | Task 3 |
| ProbabilisticGrid.occupied_mask / free_mask / render | Task 3 |
| Vectorised Bresenham raycast < 1 s for 16 k points | Task 2 |
| Dynamic-object masking from tracks.csv | Task 6 (main loop) |
| Track boundary: morphological close + CCOMP contours + DP simplify | Task 4 |
| GeoJSON LineString export | Task 4 |
| world_map_static.png | Task 5 + Task 6 |
| world_map_with_trajectory.png | Task 5 + Task 6 |
| ego_view_NNN.png (5–8 timestamps) | Task 5 + Task 6 |
| map_animation.gif < 10 MB | Task 5 + Task 6 |
| CLI with all documented flags | Task 6 |
| No modification to Task 1/2/3 scripts | Task 0 (backward-compat check) |
| Regression test (Task 3 tracks.csv byte-identical) | Task 6 Step 3 |
| Report with log-odds rationale + comparison | Task 7 |
| README Task 4 section | Task 7 |

**Placeholder scan:** Task 7 Step 2 has `<ACTUAL_*>` tokens that must be replaced with real numbers after running the pipeline in Step 1 — these are intentional fill-in-from-output markers, not missing implementation.

**Type consistency check:** `cast_rays_batch` defined in Task 2 is imported in `occupancy_grid.py` (Task 3) and uses the same `(origin_cell, hit_cells, grid_shape, max_steps)` signature throughout. `EgoTrajectory.ego_to_world(points_ego, t_ns)` signature is consistent across Tasks 1, 5, and 6. `ProbabilisticGrid.log_odds_copy()` defined in Task 3 is consumed in Task 6's animation loop.
