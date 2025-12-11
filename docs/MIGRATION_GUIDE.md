# Migration Guide: v1.0 → v2.0

This guide helps you migrate from the old ArUco_Pose_Estimation structure to the new v2.0 organized package.

## Overview of Changes

### Structure Changes
```
OLD (v1.0):                      NEW (v2.0):
├── utils.py                     ├── src/
├── calibrate.py                 │   ├── core/
├── aruco_main_tracker.py        │   │   ├── detector.py
├── aruco_pose_pipeline.py       │   │   ├── calibration.py
├── dodecahedron_model.py        │   │   └── pose_estimator.py
├── gpt_*.py                     │   ├── models/
├── utilities/                   │   │   └── dodecahedron.py
│   ├── optical_flow.py          │   ├── tracking/
│   └── dense_refinement.py      │   │   ├── optical_flow.py
└── config.yaml                  │   │   ├── refinement.py
                                 │   │   └── kalman_filter.py
                                 │   └── utils/
                                 │       ├── config.py
                                 │       ├── visualization.py
                                 │       ├── performance.py
                                 │       ├── logger.py
                                 │       └── io.py
                                 ├── scripts/
                                 │   ├── aruco_cli.py
                                 │   └── generate_markers.py
                                 ├── examples/
                                 └── tests/
```

## Step-by-Step Migration

### 1. Install New Dependencies

```bash
pip install click scipy pytest
```

### 2. Update Imports in Your Code

#### OLD:
```python
from utils import ARUCO_DICT, pose_estimation_single_camera
from dodecahedron_model import CleanDodecahedronModel
from aruco_pose_pipeline import ArucoPoseEstimator
```

#### NEW:
```python
from src.core.detector import UnifiedArucoDetector, ARUCO_DICT
from src.models.dodecahedron import CleanDodecahedronModel
from src.core.pose_estimator import MultiMarkerPoseEstimator
from src.utils.visualization import ArucoVisualizer
from src.utils.config import ProjectConfig
```

### 3. Update Configuration Files

#### OLD config.yaml:
```yaml
camera0: 0
frame_width: 1920
frame_height: 1080
aruco_size: 0.015
```

#### NEW config.yaml:
```yaml
camera:
  device_id: 0
  width: 1920
  height: 1080
  fps: 30

calibration:
  checkerboard_rows: 5
  checkerboard_cols: 7
  square_size: 0.0319
  n_frames: 10

aruco:
  dict_type: "DICT_5X5_100"
  marker_size: 0.015
```

Or use automatic conversion:
```python
from src.utils.config import ProjectConfig

# Load old format
config = ProjectConfig.from_legacy_yaml("config.yaml")

# Save as new format
config.to_yaml("config_new.yaml")
```

### 4. Update Detector Usage

#### OLD:
```python
# Version-specific code
if Version(cv2.__version__) >= Version("4.7.0"):
    arucoDict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
else:
    arucoDict = cv2.aruco.Dictionary_get(aruco_dict_type)
    arucoParams = cv2.aruco.DetectorParameters_create()
```

#### NEW:
```python
# Automatic version handling
from src.core.detector import UnifiedArucoDetector

detector = UnifiedArucoDetector("DICT_5X5_100")
corners, ids, rejected = detector.detect(image)
```

### 5. Update Calibration Workflow

#### OLD:
```python
from utils import save_frames_single_camera, calibrate_single_camera

save_frames_single_camera(calib, visualize=True)
ret, mtx, dist = calibrate_single_camera(calib, visualize=True)
```

#### NEW:
```python
# Use CLI
python scripts/aruco_cli.py calibrate --config config.yaml --visualize

# Or use API (backwards compatible)
from utils import save_frames_single_camera, calibrate_single_camera
# ... same as before
```

### 6. Update Visualization

#### OLD:
```python
cv2.aruco.drawDetectedMarkers(frame, corners, ids)
cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.05)
```

#### NEW:
```python
from src.utils.visualization import ArucoVisualizer

viz = ArucoVisualizer(K, D)
frame = viz.draw_markers(frame, corners, ids, draw_axes=True, marker_size=0.05)
frame = viz.draw_pose_info(frame, rvec, tvec)
```

### 7. Add Performance Monitoring

#### NEW Feature:
```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor()

while True:
    monitor.start_frame()
    
    # Your detection code
    corners, ids, _ = detector.detect(frame)
    monitor.record_detection(len(ids) if ids is not None else 0)
    
    monitor.end_frame()
    
    # Get stats
    stats = monitor.get_stats()
    print(f"FPS: {stats['fps']:.1f}")
```

### 8. Add Data Recording

#### NEW Feature:
```python
from src.utils.io import PoseRecorder

recorder = PoseRecorder("poses.json")

while capturing:
    # ... detection and pose estimation ...
    recorder.record_frame(time.time(), rvec, tvec, ids)

recorder.save()
```

## Common Migration Patterns

### Pattern 1: Simple Detection Script

**OLD:**
```python
import cv2
from utils import ARUCO_DICT

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT["DICT_5X5_100"])
# ... more setup ...
```

**NEW:**
```python
from src.core.detector import UnifiedArucoDetector

detector = UnifiedArucoDetector("DICT_5X5_100")
corners, ids, rejected = detector.detect(frame)
```

### Pattern 2: Pose Estimation

**OLD:**
```python
from utils import pose_estimation_single_camera

frame, markers = pose_estimation_single_camera(
    frame, "DICT_5X5_100", 0.05, K, D, show_text=True
)
```

**NEW:**
```python
from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer

detector = UnifiedArucoDetector("DICT_5X5_100")
viz = ArucoVisualizer(K, D)

corners, ids, _ = detector.detect(frame)
frame = viz.draw_markers(frame, corners, ids, draw_axes=True, marker_size=0.05)

if ids is not None:
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.05, K, D)
    frame = viz.draw_pose_info(frame, rvec, tvec)
```

### Pattern 3: Dodecahedron Tracking

**OLD:**
```python
from dodecahedron_model import CleanDodecahedronModel
from aruco_pose_pipeline import ArucoPoseEstimator

model = CleanDodecahedronModel(edge_length=0.025)
estimator = ArucoPoseEstimator(K, D, tag_data_path="aruco_group.json")
```

**NEW:**
```python
# Keep using your existing files - they're still compatible!
# Or move them to src/models/ for better organization

from src.models.dodecahedron import CleanDodecahedronModel
from aruco_pose_pipeline import ArucoPoseEstimator  # Still works

model = CleanDodecahedronModel(edge_length=0.025)
estimator = ArucoPoseEstimator(K, D, tag_data_path="aruco_group.json")
```

## Using the CLI

The new CLI provides convenient commands:

```bash
# Generate markers
python scripts/aruco_cli.py generate 0 1 2 3 --sheet -o markers/sheet.png

# Calibrate camera
python scripts/aruco_cli.py calibrate --config config.yaml --visualize

# Track markers
python scripts/aruco_cli.py track --source 0 --show-axes --record poses.json

# Analyze video
python scripts/aruco_cli.py analyze video.mp4 --calibration camera_intrinsics.npy \
    --distortion camera_distortion.npy
```

## Backward Compatibility

### Your Old Scripts Still Work!

The old `utils.py`, `calibrate.py`, and other files are **still functional**. You can:

1. **Gradual Migration**: Keep using old scripts while slowly adopting new features
2. **Hybrid Approach**: Use new utilities (logging, monitoring) with old code
3. **Full Migration**: Eventually move all code to new structure

### Example: Using New Features with Old Code

```python
# Your old imports still work
from utils import pose_estimation_single_camera

# Add new features
from src.utils.performance import PerformanceMonitor
from src.utils.logger import setup_logger

logger = setup_logger("my_tracker")
monitor = PerformanceMonitor()

# Mix old and new
while True:
    monitor.start_frame()
    
    # Old function call
    frame, markers = pose_estimation_single_camera(frame, "DICT_5X5_100", 0.05, K, D)
    
    monitor.end_frame()
    logger.info(f"FPS: {monitor.get_stats()['fps']:.1f}")
```

## Testing Your Migration

Run example scripts to verify everything works:

```bash
# Test basic detection
python examples/basic_detection.py

# Test pose estimation
python examples/pose_estimation_demo.py

# Test configuration
python examples/config_example.py
```

## Getting Help

If you encounter issues:

1. Check the examples in `examples/` directory
2. Review the new README_v2.md
3. Look at the source code documentation
4. Open an issue on GitHub

## Summary

✅ **What Changed:**
- Better code organization (src/ structure)
- Unified detector for all OpenCV versions
- Professional visualization tools
- Configuration management with validation
- Performance monitoring
- Data recording utilities
- CLI interface

✅ **What's Compatible:**
- All your existing scripts still work
- Old config.yaml can be auto-converted
- Calibration files are compatible
- ArUco detection results are identical

✅ **What's Better:**
- Cleaner imports and API
- Better error handling
- Comprehensive logging
- Performance tracking
- More examples and documentation
