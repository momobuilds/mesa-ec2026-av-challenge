# Demo Video Pipeline

Generate a recorded MP4 from the existing Task 1-7 outputs:

```bash
uv run python demo/generate_demo_video.py \
    --outputs-dir outputs \
    --bag-dir data \
    --output demo/demo_task1_to_7.mp4
```

The default video uses rendered PNG/CSV artifacts and writes:

- `demo/demo_task1_to_7.mp4`

Optional raw camera panel from the ROS bag:

```bash
uv run python demo/generate_demo_video.py \
    --bag data \
    --raw-source bag \
    --camera-topic /sensor/camera/camera_fl/image/compressed
```

Other useful controls:

```bash
uv run python demo/generate_demo_video.py --frames 80 --fps 8
uv run python demo/generate_demo_video.py --width 1280 --height 720
uv run python demo/generate_demo_video.py --output demo/custom_demo.mp4
```

The script degrades gracefully if a sensor, CSV, or rendered task image is
missing. Missing panels are replaced with placeholders while available task
outputs continue to render.
