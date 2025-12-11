# Project Organization

Clean, logical structure for the ArUco Pose Estimation package.

## 📁 Directory Structure

```
ArUco_Pose_Estimation/
├── src/                          # Source code (all utilities here!)
│   ├── core/                     # Core detection functionality
│   │   └── detector.py          # UnifiedArucoDetector
│   ├── models/                   # 3D models (dodecahedron, etc.)
│   ├── tracking/                 # Tracking algorithms
│   │   ├── optical_flow.py      # Lucas-Kanade tracking
│   │   └── dense_refinement.py  # Dense pose refinement
│   └── utils/                    # Utility functions
│       ├── config.py            # Configuration management
│       ├── io.py                # Video/data I/O
│       ├── logger.py            # Logging setup
│       ├── marker_generator.py  # Marker generation
│       ├── performance.py       # Performance monitoring
│       └── visualization.py     # Visualization tools
│
├── examples/                     # Example scripts organized by use case
│   ├── 01_basic/                # Getting started
│   ├── 02_single_marker/        # Single marker pose
│   ├── 03_multi_marker/         # Multi-marker fusion
│   ├── 04_custom_models/        # Dodecahedron tracking
│   └── 05_advanced/             # Advanced techniques
│
├── tools/                        # Command-line tools
│   ├── cli.py                   # Main CLI interface
│   └── visualize_aruco_extrinsics.py  # 3D visualization
│
├── tests/                        # Unit tests
│   ├── test_installation.py     # Installation verification
│   ├── test_dodecahedron_model.py  # Model tests
│   └── test_visualization.py    # Visualization tests
│
└── docs/                         # Documentation
    ├── README_v2.md             # Main documentation
    ├── QUICK_START.md           # Quick start guide
    ├── MIGRATION_GUIDE.md       # Migration guide
    └── ...
```

## 🎯 Design Principles

### 1. **src/ contains ALL utilities**
No more scattered utility folders. Everything is in `src/`:
- Core functionality → `src/core/`
- Tracking algorithms → `src/tracking/`
- Utility functions → `src/utils/`
- 3D models → `src/models/`

### 2. **examples/ for learning**
Organized by complexity (01→05) and use case:
- Each directory has comprehensive README
- Copy of relevant code for standalone execution
- Clear progression from basic to advanced

### 3. **tools/ for CLI only**
Command-line interface and standalone tools:
- `cli.py` - Main interface (calibrate, track, generate, analyze)
- `visualize_aruco_extrinsics.py` - 3D scene visualization

### 4. **tests/ for testing**
Standard Python testing structure:
- Unit tests for core functionality
- Installation verification
- All tests in one place

## ✅ What Changed

### Removed Directories
- ❌ `utilities/` - Moved to `src/tracking/`
- ❌ `scripts/` - Consolidated into `tools/` and `src/utils/`

### Consolidated Files
- `utilities/optical_flow.py` → `src/tracking/optical_flow.py`
- `utilities/dense_refinement.py` → `src/tracking/dense_refinement.py`
- `scripts/generate_markers.py` → `src/utils/marker_generator.py`
- `scripts/aruco_cli.py` → (removed, duplicate of `tools/cli.py`)

### Updated Imports
All examples now use:
```python
from src.tracking import OpticalFlowTracker, DenseRefiner
from src.utils.marker_generator import generate_aruco_marker
from src.utils import ProjectConfig, setup_logger
```

## 📦 Package Installation

Install in development mode to use `src/` modules anywhere:

```bash
pip install -e .
```

Now you can import from anywhere:
```python
from src.core import UnifiedArucoDetector
from src.tracking import OpticalFlowTracker
from src.utils import ProjectConfig
```

## 🚀 Usage Patterns

### Using Core Utilities
```python
from src.core import UnifiedArucoDetector
from src.utils import ProjectConfig, setup_logger

config = ProjectConfig.from_yaml("config.yaml")
detector = UnifiedArucoDetector(config.aruco.dictionary)
logger = setup_logger("my_app")
```

### Using Tracking Utilities
```python
from src.tracking import OpticalFlowTracker, DenseRefiner

tracker = OpticalFlowTracker()
refiner = DenseRefiner()
```

### Using CLI Tools
```bash
# Generate markers
python tools/cli.py generate 0 1 2 --size 200

# Calibrate camera
python tools/cli.py calibrate --rows 9 --cols 6

# Track markers
python tools/cli.py track --config config.yaml
```

### Running Examples
```bash
# Basic detection
cd examples/01_basic
python basic_detection.py

# Single marker pose
cd examples/02_single_marker
python simple_pose_estimation.py

# Advanced tracking
cd examples/05_advanced
python optical_flow.py
```

## 📖 Finding What You Need

| I want to... | Look in... |
|--------------|------------|
| Detect ArUco markers | `src/core/detector.py` |
| Configure the system | `src/utils/config.py` |
| Log messages | `src/utils/logger.py` |
| Visualize results | `src/utils/visualization.py` |
| Track with optical flow | `src/tracking/optical_flow.py` |
| Refine poses | `src/tracking/dense_refinement.py` |
| Generate markers | `src/utils/marker_generator.py` |
| Use CLI interface | `tools/cli.py` |
| Learn basics | `examples/01_basic/` |
| Learn pose estimation | `examples/02_single_marker/` |
| Track custom models | `examples/04_custom_models/` |
| Run tests | `tests/` |

## 🔧 Development Workflow

### Adding New Features

1. **Utility function** → Add to appropriate `src/` module
   - Detection/dictionary related → `src/core/`
   - Tracking algorithm → `src/tracking/`
   - Helper function → `src/utils/`
   - 3D model → `src/models/`

2. **CLI command** → Add to `tools/cli.py`

3. **Example** → Add to appropriate `examples/XX_category/`

4. **Test** → Add to `tests/`

### No More Confusion!

Before:
- "Should this go in `utilities/` or `src/utils/`?"
- "Is this a `script/` or `tools/` or `examples/`?"
- "Where do I find the optical flow code?"

After:
- All source code → `src/`
- All examples → `examples/`
- All CLI tools → `tools/`
- All tests → `tests/`

## 🎓 Benefits

1. **Clear separation of concerns**
   - Source code in `src/`
   - Examples in `examples/`
   - Tools in `tools/`
   - Tests in `tests/`

2. **Easy to find things**
   - Want utility? Check `src/utils/`
   - Want tracking algorithm? Check `src/tracking/`
   - Want example? Check `examples/`

3. **Standard Python structure**
   - Follows Python packaging conventions
   - Works with `pip install -e .`
   - Clean imports

4. **Maintainable**
   - One place for each type of code
   - No duplication
   - Easy to extend

## 📝 Migration Notes

If you have old code importing from removed directories:

### Old imports:
```python
from utilities.optical_flow import OpticalFlowTracker
from scripts.generate_markers import generate_aruco_marker
```

### New imports:
```python
from src.tracking import OpticalFlowTracker
from src.utils.marker_generator import generate_aruco_marker
```

All examples have been updated to use the new structure!
