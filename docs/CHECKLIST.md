# 🚀 ArUco Pose Estimation v2.0 - Getting Started Checklist

## ✅ Installation Checklist

### 1. Install Dependencies
```bash
cd ArUco_Pose_Estimation
pip install -r requirements.txt
pip install click scipy pytest
```

### 2. Verify Installation
```bash
python tests/test_installation.py
```

If all tests pass, you're ready to go! 🎉

---

## 📝 Quick Reference

### Common Commands

```bash
# Generate markers
python scripts/generate_markers.py --ids 0 1 2 3 --sheet -o markers/sheet.png

# Run basic detection
python examples/basic_detection.py

# Calibrate camera (requires checkerboard)
python scripts/aruco_cli.py calibrate --config config.yaml --visualize

# Track with full features
python scripts/aruco_cli.py track --source 0 --show-axes --show-stats --record data.json
```

### File Locations

| What | Where |
|------|-------|
| Source code | `src/` |
| Scripts/CLI | `scripts/` |
| Examples | `examples/` |
| Tests | `tests/` |
| Documentation | `*.md` files |
| Configuration | `config.yaml` |
| Generated markers | `markers/` (after generation) |
| Calibration output | `camera_intrinsics.npy`, `camera_distortion.npy` |

### Key Imports

```python
# Detection
from src.core.detector import UnifiedArucoDetector

# Visualization  
from src.utils.visualization import ArucoVisualizer

# Configuration
from src.utils.config import ProjectConfig

# Monitoring
from src.utils.performance import PerformanceMonitor

# Logging
from src.utils.logger import setup_logger

# I/O
from src.utils.io import VideoHandler, PoseRecorder
```

---

## 📚 Documentation Guide

Read in this order:

1. **QUICK_START.md** ← Start here! (5 minutes)
2. **README_v2.md** ← Comprehensive guide
3. **MIGRATION_GUIDE.md** ← If you have old code
4. **IMPLEMENTATION_SUMMARY.md** ← What's new and why

---

## 🎯 Your First Session

### Step 1: Verify (2 minutes)
```bash
python tests/test_installation.py
```

### Step 2: Generate Markers (2 minutes)
```bash
python scripts/generate_markers.py --ids 0 1 2 3 --sheet -o markers/sheet.png
```
Print `markers/sheet.png`

### Step 3: Test Detection (3 minutes)
```bash
python examples/basic_detection.py
```
Hold printed markers in front of camera

### Step 4: Calibrate (5 minutes)
Print a checkerboard from https://markhedleyjones.com/projects/calibration-checkerboard-collection
```bash
python scripts/aruco_cli.py calibrate --config config.yaml --visualize
```

### Step 5: Full Tracking (2 minutes)
```bash
python examples/pose_estimation_demo.py
```

**Total time: ~15 minutes to full functionality!**

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
pip install click scipy
```

### "Camera not found"
```bash
# Test camera
python -c "import cv2; print('OK' if cv2.VideoCapture(0).isOpened() else 'FAIL')"

# Try different camera index (1, 2, etc.)
```

### "No markers detected"
- Ensure good lighting
- Print markers clearly (≥200x200 pixels)
- Use correct dictionary type
- Hold marker steady and flat

### "Calibration files not found"
```bash
# Run calibration first
python scripts/aruco_cli.py calibrate --visualize
```

---

## 💡 Tips

1. **Start simple**: Run examples first, then customize
2. **Good lighting**: Essential for reliable detection
3. **Print quality**: High-quality marker prints = better tracking
4. **Calibration**: Always calibrate for accurate pose estimation
5. **Monitor performance**: Use `PerformanceMonitor` to track FPS

---

## 🎓 Learning Path

### Beginner:
1. Run `basic_detection.py`
2. Generate and detect your own markers
3. Read `QUICK_START.md`

### Intermediate:
1. Calibrate your camera
2. Run `pose_estimation_demo.py`
3. Experiment with CLI commands
4. Read `README_v2.md`

### Advanced:
1. Use API in your own code
2. Add custom models to `src/models/`
3. Extend tracking in `src/tracking/`
4. Read `IMPLEMENTATION_SUMMARY.md`

---

## 🔗 Resources

- **Checkerboard Patterns**: https://markhedleyjones.com/projects/calibration-checkerboard-collection
- **OpenCV ArUco Docs**: https://docs.opencv.org/4.x/d9/d6d/tutorial_table_of_content_aruco.html
- **Calibration Tutorial**: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

---

## 📧 Get Help

- Check documentation files (4 comprehensive guides)
- Run verification: `python tests/test_installation.py`
- Review examples in `examples/`
- Open GitHub issue for bugs

---

## 🎉 You're Ready!

Your ArUco Pose Estimation toolkit is now:
- ✅ Properly structured
- ✅ Well documented  
- ✅ Feature-rich
- ✅ Production-ready
- ✅ Easy to use
- ✅ Easy to extend

**Start with:** `python examples/basic_detection.py`

Happy tracking! 🚀
