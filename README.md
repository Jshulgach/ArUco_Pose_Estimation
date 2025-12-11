# ArUco Pose Estimation Toolkit v2.0

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.6+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A professional, robust toolkit for ArUco marker detection, pose estimation, and custom 3D model tracking in Python.

<div align="center">
<img src="assets/aruco-track.gif" width="700">
</div>

## ✨ Features

### Core Capabilities
- **Universal ArUco Detection** - Seamless support for OpenCV versions 4.x and 3.x
- **Camera Calibration** - Single and dual-camera calibration workflows
- **Pose Estimation** - Real-time 6DOF pose tracking from markers
- **Custom 3D Models** - Dodecahedron and custom geometry tracking
- **Multi-Marker Fusion** - Improved accuracy from marker groups
- **Performance Monitoring** - Built-in FPS and error tracking
- **Data Recording** - Export pose data in JSON/CSV formats

### Advanced Features
- **Pose Refinement** - Scipy-based optimization for sub-pixel accuracy
- **Optical Flow Tracking** - Maintain tracking between detections
- **Kalman Filtering** - Smooth pose trajectories
- **Reprojection Visualization** - Debug pose estimation quality
- **CLI Interface** - Professional command-line tools
- **Configuration Management** - YAML-based with validation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Jshulgach/ArUco_Pose_Estimation.git
cd ArUco_Pose_Estimation

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### Basic Usage

#### 1. Generate ArUco Markers

```bash
# Using the example script
python examples/01_basic/generate_markers.py --ids 0 1 2 3 4 5 --output markers/

# Or use the CLI tool
python tools/cli.py generate 0 1 2 3 4 5 --size 200 --dict DICT_4X4_50
```

#### 2. Camera Calibration

```bash
# Using the calibration example
python examples/01_basic/camera_calibration.py

# Or use the CLI
python tools/cli.py calibrate --rows 9 --cols 6 --output calibration/
```

#### 3. Real-time Tracking

```bash
# Simple pose estimation
python examples/02_single_marker/simple_pose_estimation.py

# Or use advanced tracking demo
python examples/02_single_marker/marker_tracking.py
```

### Python API Example

```python
from src.core.detector import UnifiedArucoDetector
from src.utils.visualization import ArucoVisualizer
from src.utils.performance import PerformanceMonitor
import cv2
import numpy as np

# Load calibration
K = np.load("camera_intrinsics.npy")
D = np.load("camera_distortion.npy")

# Initialize components
detector = UnifiedArucoDetector("DICT_5X5_100")
visualizer = ArucoVisualizer(K, D)
monitor = PerformanceMonitor()

# Capture and process video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect markers
    corners, ids, rejected = detector.detect(frame)
    
    # Draw visualization
    frame = visualizer.draw_markers(frame, corners, ids, 
                                    draw_axes=True, marker_size=0.05)
    
    # Track performance
    monitor.record_detection(len(ids) if ids is not None else 0)
    
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print statistics
print(monitor.get_summary())
```

## 📁 Project Structure

```
ArUco_Pose_Estimation/
├── src/                   # Source code
│   ├── core/             # Core detection and calibration
│   │   └── detector.py   # Unified ArUco detector
│   ├── models/           # 3D geometric models
│   │   ├── dodecahedron_model.py
│   │   ├── aruco_pose_pipeline.py
│   │   └── aruco_pose_estimator.py
│   ├── tracking/         # Advanced tracking utilities
│   │   ├── optical_flow.py
│   │   └── dense_refinement.py
│   └── utils/            # Utilities
│       ├── config.py
│       ├── visualization.py
│       ├── performance.py
│       ├── marker_generator.py
│       └── io.py
├── examples/             # Learning examples (01→05)
│   ├── 01_basic/        # Calibration, marker generation
│   ├── 02_single_marker/ # Single marker tracking
│   ├── 03_multi_marker/ # Multi-marker fusion
│   ├── 04_custom_models/ # Dodecahedron tracking
│   └── 05_advanced/     # Advanced techniques
├── tools/               # CLI and visualization tools
│   └── cli.py          # Main command-line interface
├── tests/              # Unit tests
├── docs/               # Documentation
├── config/             # Configuration files
├── assets/             # Media files
└── setup.py           # Package setup
```

## 🎯 Usage Examples

### Command-Line Interface

The toolkit includes a comprehensive CLI:

```bash
# Calibration
python tools/cli.py calibrate --rows 9 --cols 6 --output calibration/

# Real-time tracking
python tools/cli.py track --source 0 --config config.yaml --output poses.json

# Generate markers
python tools/cli.py generate 0 1 2 3 4 5 --size 200 --dict DICT_4X4_50

# Analyze recorded data
python tools/cli.py analyze --input tracking.json --plot --report
```

### Configuration Management

```python
from src.utils.config import ProjectConfig

# Load configuration
config = ProjectConfig.from_yaml("config.yaml")

# Validate
config.validate()

# Access settings
print(f"Camera: {config.camera.width}x{config.camera.height}")
print(f"Marker size: {config.aruco.marker_size} m")

# Save configuration
config.to_yaml("output_config.yaml")
```

### Performance Monitoring

```python
from src.utils.performance import PerformanceMonitor

monitor = PerformanceMonitor(window_size=30)

# Track a frame
monitor.start_frame()
# ... do detection and pose estimation ...
monitor.record_detection(num_markers=5)
monitor.end_frame()

# Get statistics
stats = monitor.get_stats()
print(f"FPS: {stats['fps']:.1f}")
print(f"Avg markers: {stats['avg_markers']:.1f}")

# Print summary
print(monitor.get_summary())
```

### Data Recording

```python
from src.utils.io import PoseRecorder, VideoHandler
import time

# Record pose data
recorder = PoseRecorder("poses.json", save_interval=100)

with VideoHandler(source=0) as video:
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # ... detect and estimate pose ...
        
        recorder.record_frame(
            timestamp=time.time(),
            rvec=rvec,
            tvec=tvec,
            marker_ids=ids,
            reprojection_error=error
        )

recorder.save()
stats = recorder.get_statistics()
print(f"Recorded {stats['total_frames']} frames")
```

## 🔧 Configuration

Example `config.yaml`:

```yaml
camera:
  device_id: 0
  width: 1920
  height: 1080
  fps: 30

calibration:
  checkerboard_rows: 5
  checkerboard_cols: 7
  square_size: 0.0319  # meters
  n_frames: 10
  img_dir: "frames"
  cooldown: 100
  view_resize: 2.0

aruco:
  dict_type: "DICT_5X5_100"
  marker_size: 0.015  # meters
```

## 📊 Performance

Typical performance on modern hardware:
- **Detection**: 50-100 FPS (single marker)
- **Pose Estimation**: 40-80 FPS
- **Reprojection Error**: < 1 pixel (calibrated camera)
- **Tracking Latency**: < 20ms

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📝 Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{aruco_pose_estimation,
  author = {Shulgach, Jonathan},
  title = {ArUco Pose Estimation Toolkit},
  year = {2025},
  url = {https://github.com/Jshulgach/ArUco_Pose_Estimation}
}
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Quick Start Guide](docs/QUICK_START.md)** - Get started in 5 minutes
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade from legacy code
- **[Examples Guide](examples/README.md)** - Complete examples walkthrough
- **[Tools Reference](tools/README.md)** - CLI and visualization tools
- **[Implementation Details](docs/IMPLEMENTATION_SUMMARY.md)** - Technical deep dive

See [docs/README.md](docs/README.md) for the complete documentation index.

## 🙏 Acknowledgements

This project builds upon:
- [GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python](https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python) - Initial inspiration
- OpenCV ArUco module documentation
- Community contributions and feedback

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Camera not detected
```bash
# List available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(10)])"
```

### Calibration files not found
Make sure to run calibration first:
```bash
python examples/01_basic/camera_calibration.py
```

### Poor tracking performance
- Ensure proper lighting
- Check marker print quality
- Verify calibration accuracy
- Adjust detector parameters

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📧 Contact

For questions or issues:
- **Author**: Jonathan Shulgach
- **Email**: jshulgac@andrew.cmu.edu
- **Issues**: [GitHub Issues](https://github.com/Jshulgach/ArUco_Pose_Estimation/issues)

---

⭐ If you find this project useful, please star it on GitHub!
