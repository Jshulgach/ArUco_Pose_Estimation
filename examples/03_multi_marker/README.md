# Multi-Marker Pose Estimation

This directory contains examples for tracking multiple ArUco markers simultaneously and fusing their poses for improved accuracy.

## Files

### 1. `multi_marker_fusion.py`
Advanced pose estimation using multiple markers with pose fusion.

**Usage:**
```bash
python multi_marker_fusion.py
```

**What it does:**
- Detects multiple ArUco markers simultaneously
- Fuses pose estimates from multiple markers
- Computes single, more accurate pose estimate
- Handles partial marker occlusions
- Visualizes all detected markers

**Key Features:**
- **ArucoPoseEstimator class:** Core multi-marker fusion logic
- **Marker grouping:** Define groups of markers that belong together
- **Weighted averaging:** More visible markers get higher weight
- **Occlusion handling:** Works even when some markers are hidden
- **Reprojection validation:** Rejects poor pose estimates

**Prerequisites:**
- Calibrated camera
- Multiple ArUco markers (3+ recommended)
- Marker group configuration file (JSON)

---

## Why Use Multiple Markers?

### Advantages

1. **Higher accuracy:** Averaging multiple poses reduces noise
2. **Occlusion robustness:** Object still tracked if some markers hidden
3. **Larger tracking volume:** Markers placed on different faces
4. **Better angle coverage:** Some markers always well-visible
5. **Redundancy:** System degrades gracefully if markers lost

### Use Cases

- **Object tracking:** Attach markers to different sides of object
- **Robot pose estimation:** Multiple markers on robot body
- **Room-scale tracking:** Markers distributed in environment
- **Rigid body tracking:** Markers on known configuration

---

## Multi-Marker Pose Fusion Algorithm

### Step 1: Detect All Markers
```python
markers, ids = detector.detect(frame)
```

### Step 2: Estimate Individual Poses
For each marker:
```python
rvec, tvec = cv2.solvePnP(marker_3d_points, marker_2d_corners)
```

### Step 3: Transform to Common Reference
Convert all marker poses to object center:
```python
object_pose = marker_pose @ marker_to_object_transform
```

### Step 4: Fuse Poses
Weighted average based on:
- Reprojection error (lower = higher weight)
- Marker area (larger = higher weight)
- Viewing angle (frontal = higher weight)

### Step 5: Validate
- Check reprojection error of fused pose
- Verify pose is physically plausible
- Compare with previous pose (temporal consistency)

---

## Configuration

### Marker Group JSON Format

```json
{
  "markers": [
    {
      "id": 0,
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0],
      "size": 0.05
    },
    {
      "id": 1,
      "position": [0.1, 0.0, 0.0],
      "orientation": [0.0, 90.0, 0.0],
      "size": 0.05
    }
  ],
  "reference_frame": "marker_0"
}
```

### Parameters

- `id`: Marker ID from ArUco dictionary
- `position`: [x, y, z] in meters relative to reference frame
- `orientation`: [roll, pitch, yaw] in degrees
- `size`: Marker side length in meters
- `reference_frame`: Which marker defines origin

---

## Quick Start

### 1. Generate Multiple Markers

```bash
cd ../01_basic
python generate_markers.py --id 0 --size 200
python generate_markers.py --id 1 --size 200
python generate_markers.py --id 2 --size 200
cd ../03_multi_marker
```

### 2. Print and Attach Markers

- Print markers at known size (measure with ruler!)
- Attach to object at known relative positions
- Measure distances between markers accurately

### 3. Create Configuration

Create `marker_group.json` with marker positions:
```json
{
  "markers": [
    {"id": 0, "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0], "size": 0.05},
    {"id": 1, "position": [0.1, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0], "size": 0.05},
    {"id": 2, "position": [0.05, 0.1, 0.0], "orientation": [0.0, 0.0, 0.0], "size": 0.05}
  ]
}
```

### 4. Run Multi-Marker Tracking

```bash
python multi_marker_fusion.py --config marker_group.json
```

---

## Tips for Best Results

### Marker Placement

1. **Spacing:** Spread markers as far apart as practical
2. **Orientation:** Place on different faces for better viewing angles
3. **Redundancy:** Use 4-6 markers for robust tracking
4. **Avoid clustering:** Don't place markers too close together

### Measurement

1. **Accurate distances:** Use calipers or ruler for precise measurements
2. **Check alignment:** Ensure markers are flat and properly aligned
3. **Reference frame:** Choose stable marker as reference (e.g., center marker)
4. **Coordinate system:** Use right-handed coordinate system

### Calibration

1. **Good camera calibration:** Critical for accurate fusion
2. **Marker size accuracy:** Measure printed markers with ruler
3. **Configuration file:** Double-check all positions and orientations
4. **Test with known setup:** Validate with easily measurable configuration

---

## Troubleshooting

### "Poses don't align correctly"
- Verify marker positions in configuration match physical setup
- Check coordinate system handedness
- Ensure camera calibration is accurate
- Measure marker sizes carefully

### "Jittery tracking"
- Increase fusion window (more markers = smoother)
- Apply temporal filtering
- Check for marker detection instability
- Improve lighting conditions

### "High reprojection error"
- Recalibrate camera
- Verify marker configuration accuracy
- Check for lens distortion at edges
- Ensure markers are flat and not warped

### "Some markers not detected"
- Verify ArUco dictionary matches all markers
- Check lighting (avoid shadows on markers)
- Ensure all markers are in camera view
- Increase marker size if too small

---

## Performance Optimization

### Detection Speed
- Use smaller ArUco dictionary (e.g., DICT_4X4_50 instead of DICT_6X6_250)
- Reduce image resolution if real-time performance needed
- Limit search to expected marker IDs

### Accuracy
- Use more markers (4-6 optimal)
- Calibrate camera carefully
- Measure marker positions precisely
- Use larger markers when possible

---

## Next Steps

- Explore **custom 3D models** in `04_custom_models/` for non-planar marker arrangements
- Learn **dodecahedron tracking** for 360° coverage
- Try **advanced refinement** in `05_advanced/` for sub-pixel accuracy

---

## Technical References

**solvePnP algorithms:**
- `SOLVEPNP_ITERATIVE`: Standard iterative refinement
- `SOLVEPNP_EPNP`: Efficient PnP for 4+ points
- `SOLVEPNP_SQPNP`: Most accurate for well-conditioned problems

**Pose averaging:**
- Rotation: Use quaternion averaging or Lie algebra
- Translation: Weighted arithmetic mean
- Weights: Based on reprojection error and marker visibility
