# Quick Start Guide - ArUco Pose Estimation v2.0

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Webcam or video source
- Checkerboard pattern (optional, for calibration)

## Installation

```bash
# Navigate to the package directory
cd ArUco_Pose_Estimation

# Install dependencies
pip install -r requirements.txt

# Install additional new dependencies
pip install click scipy

# Optional: Install in development mode
pip install -e .
```

## Quick Examples

### 1. Generate ArUco Markers (30 seconds)

```bash
# Generate individual markers
python scripts/generate_markers.py --ids 0 1 2 3 4 5 --output markers/

# Generate a printable sheet with 6 markers
python scripts/generate_markers.py --ids 0 1 2 3 4 5 --sheet --output markers/sheet.png
```

**Result**: You now have ArUco markers in the `markers/` directory!

### 2. Basic Detection (1 minute)

```bash
# Run basic detection example
python examples/basic_detection.py
```

**What it does**: Opens your webcam and detects ArUco markers in real-time

**Controls**:
- Press `q` to quit

### 3. Camera Calibration (5 minutes)

You need a printed checkerboard pattern. Get one here:
https://markhedleyjones.com/projects/calibration-checkerboard-collection

```bash
# Run calibration
python scripts/aruco_cli.py calibrate --config config.yaml --visualize
```

**Steps**:
1. Hold checkerboard in front of camera
2. Press SPACE to start capturing
3. Move checkerboard to different positions/angles
4. Wait for 10 frames to be captured
5. Calibration completes automatically

**Result**: Creates `camera_intrinsics.npy` and `camera_distortion.npy`

### 4. Pose Estimation (1 minute)

**Prerequisites**: Complete step 3 (calibration)

```bash
# Run pose estimation demo
python examples/pose_estimation_demo.py
```

**What it does**: 
- Detects markers
- Estimates 6DOF pose
- Draws 3D axes on markers
- Shows position (X, Y, Z) and orientation (Pitch, Roll, Yaw)
- Displays performance stats (FPS)

**Controls**:
- Press `q` to quit
- Press `r` to reset statistics

### 5. Full Tracking with Recording (2 minutes)

```bash
# Track with all features enabled
python scripts/aruco_cli.py track \
    --source 0 \
    --show-axes \
    --show-stats \
    --record tracking_data.json \
    --output tracking_video.mp4
```

**What it does**:
- Real-time marker tracking
- 3D axes overlay
- Performance statistics
- Records pose data to JSON
- Saves output video

**Result**: You get `tracking_data.json` and `tracking_video.mp4`

## Using in Your Code

### Minimal Example (10 lines)

```python
from src.core.detector import UnifiedArucoDetector
import cv2

detector = UnifiedArucoDetector("DICT_5X5_100")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    corners, ids, _ = detector.detect(frame)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imshow('ArUco', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### With Pose Estimation (20 lines)

```python
from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer
import cv2
import numpy as np

# Load calibration
K = np.load("camera_intrinsics.npy")
D = np.load("camera_distortion.npy")

detector = UnifiedArucoDetector("DICT_5X5_100")
viz = ArucoVisualizer(K, D)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    corners, ids, _ = detector.detect(frame)
    
    # Draw markers with 3D axes
    frame = viz.draw_markers(frame, corners, ids, draw_axes=True, marker_size=0.05)
    
    # Estimate and display pose for first marker
    if ids is not None and len(ids) > 0:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, K, D)
        frame = viz.draw_pose_info(frame, rvec, tvec)
    
    cv2.imshow('ArUco Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### With Performance Monitoring (25 lines)

```python
from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer
from src.utils.performance import PerformanceMonitor
import cv2
import numpy as np

K = np.load("camera_intrinsics.npy")
D = np.load("camera_distortion.npy")

detector = UnifiedArucoDetector("DICT_5X5_100")
viz = ArucoVisualizer(K, D)
monitor = PerformanceMonitor()

cap = cv2.VideoCapture(0)

while True:
    monitor.start_frame()
    ret, frame = cap.read()
    
    corners, ids, _ = detector.detect(frame)
    monitor.record_detection(len(ids) if ids is not None else 0)
    
    frame = viz.draw_markers(frame, corners, ids, draw_axes=True, marker_size=0.05)
    
    # Show stats on screen
    stats = monitor.get_stats()
    frame = viz.draw_performance_stats(frame, {'FPS': stats['fps'], 'Markers': stats['avg_markers']})
    
    monitor.end_frame()
    cv2.imshow('ArUco', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(monitor.get_summary())
```

## Common Issues & Solutions

### Issue: "Camera not found"
```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera failed')"

# Try different camera index
python examples/basic_detection.py  # Edit line: cap = cv2.VideoCapture(1)
```

### Issue: "Calibration files not found"
```bash
# Run calibration first
python scripts/aruco_cli.py calibrate --config config.yaml --visualize
```

### Issue: "No markers detected"
- Ensure good lighting
- Print markers clearly (recommended: 200x200 pixels minimum)
- Verify using correct dictionary type
- Check if markers are in camera view

### Issue: "Poor tracking accuracy"
- Run camera calibration
- Use larger markers
- Improve lighting conditions
- Ensure markers are flat and not distorted

## Next Steps

1. **Read the full README**: See `README_v2.md` for comprehensive documentation
2. **Check examples**: Explore the `examples/` directory
3. **Read migration guide**: If you have old code, see `MIGRATION_GUIDE.md`
4. **Try advanced features**: Dodecahedron tracking, multi-marker fusion
5. **Customize**: Modify `config.yaml` for your setup

## Command Reference

```bash
# Calibration
python scripts/aruco_cli.py calibrate --config config.yaml --visualize

# Tracking
python scripts/aruco_cli.py track --source 0 --show-axes --show-stats

# Generate markers
python scripts/aruco_cli.py generate 0 1 2 3 --sheet -o markers/sheet.png

# Analyze video
python scripts/aruco_cli.py analyze video.mp4 --calibration camera_intrinsics.npy \
    --distortion camera_distortion.npy --output analyzed.mp4

# Get help
python scripts/aruco_cli.py --help
python scripts/aruco_cli.py track --help
```

## Configuration Quick Reference

Edit `config.yaml`:

```yaml
camera:
  device_id: 0          # Change camera (0, 1, 2...)
  width: 1920           # Resolution
  height: 1080
  fps: 30

aruco:
  dict_type: "DICT_5X5_100"  # Marker dictionary
  marker_size: 0.05          # Physical size in meters

calibration:
  checkerboard_rows: 5       # Inner corners
  checkerboard_cols: 7
  square_size: 0.025         # Square size in meters
  n_frames: 10               # Frames to capture
```

## Resources

- **Checkerboard Patterns**: https://markhedleyjones.com/projects/calibration-checkerboard-collection
- **ArUco Marker Info**: https://docs.opencv.org/4.x/d9/d6d/tutorial_table_of_content_aruco.html
- **OpenCV Calibration**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

## Getting Help

- Check `README_v2.md` for detailed documentation
- Review `examples/` directory for code samples
- Look at `MIGRATION_GUIDE.md` if migrating from v1.0
- Open an issue on GitHub for bugs

---

**You're ready to go! Start with example 1 and work your way up!** 🚀
