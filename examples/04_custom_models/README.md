# Custom 3D Models

This directory contains examples for tracking custom 3D shapes with ArUco markers, including the advanced dodecahedron tracking system.

## Files

### 1. `dodecahedron_model.py`
Clean implementation of the dodecahedron 3D model geometry.

**Usage:**
```python
from dodecahedron_model import CleanDodecahedronModel

model = CleanDodecahedronModel(marker_size=0.05)
vertices = model.vertices  # 20 vertices
faces = model.faces  # 12 pentagonal faces
```

**What it does:**
- Defines 20-vertex dodecahedron geometry
- Computes 12 pentagonal face centers
- Provides marker placement coordinates
- Exports model to JSON

**Key Features:**
- **Precise geometry:** Golden ratio-based dodecahedron
- **Face orientation:** Each face can hold an ArUco marker
- **Vertex ordering:** Consistent winding for face normals
- **Scale control:** Adjustable size via `marker_size` parameter

---

### 2. `dodecahedron_pose.py`
Pose estimation specifically for dodecahedron marker arrangements.

**Usage:**
```bash
python dodecahedron_pose.py
```

**What it does:**
- Detects markers on dodecahedron faces
- Estimates full 6DOF pose using multi-marker fusion
- Handles partial visibility (only 3-6 faces visible at once)
- Visualizes 3D dodecahedron overlay on video

**Prerequisites:**
- Dodecahedron with ArUco markers on faces
- Camera calibration
- `dodecahedron_with_markers.json` configuration

---

### 3. `gpt_aruco_pose_estimator.py`
ArUco pose estimator class for dodecahedron tracking.

**Usage:**
```python
from gpt_aruco_pose_estimator import ArucoPoseEstimator

estimator = ArucoPoseEstimator(
    camera_matrix, dist_coeffs,
    marker_size=0.05,
    aruco_dict=cv2.aruco.DICT_4X4_50
)
pose = estimator.estimate_pose(frame)
```

**Features:**
- Multi-marker pose fusion algorithm
- Robust outlier rejection
- Confidence scoring
- Temporal smoothing option

---

### 4. `dodecahedron_tracking_demo.py`
Complete demo of dodecahedron tracking with visualization.

**Usage:**
```bash
python dodecahedron_tracking_demo.py
```

**What it does:**
- Real-time dodecahedron pose tracking
- 3D model overlay rendering
- Performance statistics
- Recording capabilities

**Features:**
- Full v2.0 API integration
- Professional visualization
- Data logging
- Error handling

---

### 5. `generate_model.py`
Tool to generate custom 3D models with marker placements.

**Usage:**
```bash
python generate_model.py --shape dodecahedron --size 0.05
```

**Supported shapes:**
- `dodecahedron`: 12-face regular dodecahedron
- `cube`: 6-face cube
- `icosahedron`: 20-face icosahedron

**What it does:**
- Computes 3D geometry for selected shape
- Calculates marker placement on faces
- Exports to JSON format
- Generates STL for 3D printing

---

### 6. `generate_model_to_json.py`
Export dodecahedron model configuration to JSON.

**Usage:**
```bash
python generate_model_to_json.py --output dodecahedron_config.json
```

**Output format:**
```json
{
  "shape": "dodecahedron",
  "marker_size": 0.05,
  "faces": [
    {
      "id": 0,
      "center": [x, y, z],
      "normal": [nx, ny, nz],
      "vertices": [[x1,y1,z1], [x2,y2,z2], ...]
    }
  ]
}
```

---

## Why Use a Dodecahedron?

### Advantages

1. **360° coverage:** 12 faces provide markers visible from most angles
2. **No preferred orientation:** Nearly spherical symmetry
3. **Robust tracking:** Always 3-6 faces visible from any viewpoint
4. **High accuracy:** Multiple markers fused for each pose estimate
5. **Stable geometry:** Rigid 3D structure prevents deformation

### Comparison

| Shape | Faces | Coverage | Accuracy | Complexity |
|-------|-------|----------|----------|------------|
| Single marker | 1 | 180° | ★★☆☆☆ | ★☆☆☆☆ |
| Cube | 6 | 270° | ★★★☆☆ | ★★☆☆☆ |
| Dodecahedron | 12 | 360° | ★★★★★ | ★★★★☆ |
| Icosahedron | 20 | 360° | ★★★★★ | ★★★★★ |

---

## Building a Dodecahedron

### Option 1: 3D Printing

1. **Generate STL:**
   ```bash
   python generate_model.py --shape dodecahedron --size 0.05 --output dode.stl
   ```

2. **Print:** Use PLA or ABS, 20% infill, no supports needed

3. **Post-processing:** Sand faces smooth for marker adhesion

4. **Attach markers:** Print markers and glue to faces

### Option 2: Paper Craft

1. **Download template:** Use net pattern for dodecahedron
2. **Print on cardstock:** High-quality printer recommended
3. **Cut and fold:** Follow fold lines carefully
4. **Glue tabs:** Assemble into 3D shape
5. **Attach markers:** Glue markers to each face

### Option 3: Purchase

- Buy foam or wooden dodecahedron
- Measure face size
- Generate markers to fit
- Attach with adhesive

---

## Marker Placement

### Face Numbering

Faces are numbered 0-11 in consistent order:
- Face 0: Top face
- Faces 1-5: Upper ring
- Faces 6-10: Lower ring
- Face 11: Bottom face

### Orientation

Each marker should:
- Be centered on face
- Have consistent orientation (e.g., top edge parallel to a dodecahedron edge)
- Lay completely flat on face
- Not overlap face edges

### Measurement

Measure dodecahedron dimensions:
```bash
python dodecahedron_model.py --measure
```

This provides:
- Edge length
- Face center to center distance
- Inscribed sphere radius
- Suggested marker size (80% of face width)

---

## Configuration

### dodecahedron_with_markers.json

```json
{
  "model": {
    "type": "dodecahedron",
    "edge_length": 0.10,
    "marker_size": 0.04
  },
  "faces": [
    {"id": 0, "marker_id": 0, "center": [0.0, 0.0, 0.081]},
    {"id": 1, "marker_id": 1, "center": [...]}
  ],
  "tracking": {
    "min_markers": 2,
    "fusion_method": "weighted_average",
    "outlier_threshold": 3.0
  }
}
```

### Parameters

- `edge_length`: Physical edge length in meters
- `marker_size`: ArUco marker side length in meters
- `marker_id`: Which ArUco ID is on each face
- `min_markers`: Minimum markers required for pose estimate
- `fusion_method`: How to combine multiple marker poses
- `outlier_threshold`: Reprojection error threshold (pixels)

---

## Quick Start

### 1. Build Dodecahedron

Choose one method above and construct your dodecahedron with markers.

### 2. Measure Dimensions

```bash
# Measure edge length with ruler
# Measure marker size
```

### 3. Generate Configuration

```bash
python generate_model_to_json.py \
  --edge-length 0.10 \
  --marker-size 0.04 \
  --output dodecahedron_with_markers.json
```

### 4. Calibrate Camera

```bash
cd ../01_basic
python camera_calibration.py
cd ../04_custom_models
```

### 5. Run Tracking

```bash
python dodecahedron_tracking_demo.py \
  --config dodecahedron_with_markers.json
```

---

## Advanced Usage

### Custom Shapes

Create your own polyhedron:

```python
from dodecahedron_model import PolyhedronModel

# Define vertices and faces
vertices = [
    [0, 0, 1], [1, 0, 0], [0, 1, 0], ...
]
faces = [
    [0, 1, 2], [1, 3, 2], ...
]

model = PolyhedronModel(vertices, faces, marker_size=0.05)
model.export_json("custom_model.json")
```

### Multi-Object Tracking

Track multiple dodecahedra:

```python
tracker = MultiObjectTracker([
    ("dode1", dodecahedron_config_1),
    ("dode2", dodecahedron_config_2)
])

poses = tracker.track(frame)
for name, pose in poses.items():
    print(f"{name}: {pose}")
```

---

## Troubleshooting

### "Not enough markers detected"
- At least 2-3 markers must be visible
- Improve lighting
- Check marker dictionary matches configuration
- Ensure markers are not too small

### "Pose jumps or unstable"
- Increase `min_markers` threshold
- Improve marker placement flatness
- Check for marker occlusions
- Apply temporal smoothing

### "High reprojection error"
- Verify dodecahedron measurements are accurate
- Recalibrate camera
- Check marker size parameter
- Ensure dodecahedron is rigid (not deforming)

### "Markers detected but pose wrong"
- Check face numbering matches configuration
- Verify marker orientations are consistent
- Ensure coordinate system handedness is correct
- Validate model geometry export

---

## Performance

### Optimization Tips

1. **Limit marker search:** Only look for expected marker IDs
2. **Reduce resolution:** Downsample image if real-time needed
3. **GPU acceleration:** Use CUDA-enabled OpenCV
4. **Multi-threading:** Detect and estimate pose in parallel

### Benchmarks

| Resolution | Markers | FPS (CPU) | FPS (GPU) |
|------------|---------|-----------|-----------|
| 640×480 | 3-4 | ~60 | ~120 |
| 1280×720 | 3-4 | ~30 | ~80 |
| 1920×1080 | 3-4 | ~15 | ~45 |

---

## Next Steps

- Try **advanced refinement** in `05_advanced/` for sub-pixel accuracy
- Combine with **optical flow** for smooth inter-frame tracking
- Explore **predictive filtering** for handling fast motion
- Add **pose refinement** with ICP or dense alignment

---

## References

- **Dodecahedron geometry:** Regular polyhedron with 12 pentagonal faces
- **Multi-marker fusion:** Weighted averaging of individual marker poses
- **Outlier rejection:** RANSAC-like approach to remove bad markers
- **Coordinate frames:** Right-handed system with dodecahedron center as origin
