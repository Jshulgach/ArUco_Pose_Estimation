# Single Marker Pose Estimation

This directory contains examples for estimating the 3D pose (position and orientation) of individual ArUco markers.

## Files

### 1. `simple_pose_estimation.py`
Basic pose estimation for a single ArUco marker.

**Usage:**
```bash
python simple_pose_estimation.py
```

**What it does:**
- Detects single ArUco marker
- Estimates 6DOF pose (x, y, z, roll, pitch, yaw)
- Draws 3D coordinate axes on marker
- Shows real-time pose data

**Prerequisites:**
- Calibrated camera (run `01_basic/camera_calibration.py` first)
- `calibration_matrix.npy` and `distortion_coefficients.npy` in project root
- Printed ArUco marker with known physical size

**Configuration:**
- Edit `marker_size` parameter (in meters) to match your printed marker

---

### 2. `marker_tracking.py`
Enhanced marker tracking with the main tracking system.

**Usage:**
```bash
python marker_tracking.py
```

**What it does:**
- Real-time marker detection and pose tracking
- Performance monitoring and FPS display
- Visualization of marker coordinate frames
- Reprojection error visualization
- Data recording capabilities

**Features:**
- Uses `UnifiedArucoDetector` for cross-version compatibility
- `ArucoVisualizer` for professional overlays
- `PerformanceMonitor` for metrics tracking
- Configurable via YAML

**Prerequisites:**
- Camera calibration files
- `config.yaml` in project root

---

### 3. `pose_estimation_demo.py`
Comprehensive demo using the new v2.0 utilities.

**Usage:**
```bash
python pose_estimation_demo.py
```

**What it does:**
- Demonstrates full v2.0 API usage
- Shows configuration loading from YAML
- Visualizes pose with professional graphics
- Records pose data to JSON/CSV
- Performance statistics display

**Features:**
- Type-safe configuration management
- Automatic data recording
- Real-time visualization
- Error handling and logging

---

## Understanding Pose Estimation

### What is 6DOF Pose?

A marker's pose consists of:
- **Translation** (x, y, z): Position in 3D space relative to camera
- **Rotation** (roll, pitch, yaw): Orientation around each axis

### Coordinate System

The marker coordinate system:
- **X-axis (Red):** Points right across the marker
- **Y-axis (Green):** Points down the marker
- **Z-axis (Blue):** Points out perpendicular to marker surface

### Accuracy Factors

1. **Calibration quality:** Better calibration = more accurate pose
2. **Marker size:** Larger markers provide more stable pose estimates
3. **Distance:** Closer markers give better accuracy
4. **Viewing angle:** Frontal views are more accurate than oblique angles
5. **Marker detection:** Clean, well-lit markers with sharp corners

---

## Quick Start

1. **Ensure camera is calibrated:**
   ```bash
   cd ../01_basic
   python camera_calibration.py
   cd ../02_single_marker
   ```

2. **Run simple pose estimation:**
   ```bash
   python simple_pose_estimation.py
   ```

3. **Try enhanced tracking:**
   ```bash
   python marker_tracking.py
   ```

4. **Explore v2.0 demo:**
   ```bash
   python pose_estimation_demo.py
   ```

---

## Troubleshooting

### "Calibration files not found"
Run calibration first: `cd ../01_basic && python camera_calibration.py`

### "Pose estimation unstable"
- Check marker size parameter matches physical marker
- Ensure marker is flat and not curved/wrinkled
- Improve lighting conditions
- Get closer to the marker
- Recalibrate camera if drift persists

### "No markers detected"
- Verify marker dictionary matches (e.g., DICT_4X4_50)
- Check lighting (avoid glare and shadows)
- Print marker at higher resolution
- Ensure marker is in camera view and not too small

---

## Next Steps

- Try **multi-marker tracking** in `03_multi_marker/` for improved accuracy
- Explore **custom 3D models** in `04_custom_models/` for dodecahedron tracking
- Learn **advanced techniques** in `05_advanced/` for optical flow and refinement

---

## Technical Details

### Pose Estimation Algorithm

1. **Detect marker corners** using ArUco detector
2. **Solve PnP (Perspective-n-Point)** problem:
   - Input: 2D image corners + 3D marker model
   - Output: Rotation vector (rvec) and translation vector (tvec)
3. **Convert rotation** from Rodrigues to Euler angles
4. **Transform coordinates** to desired reference frame

### Reprojection Error

The reprojection error measures pose quality:
- **Good:** < 1.0 pixel
- **Acceptable:** 1.0 - 3.0 pixels
- **Poor:** > 3.0 pixels (check calibration or marker quality)

Lower reprojection error = more accurate pose estimate.
