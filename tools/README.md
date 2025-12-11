# Tools

Command-line utilities and helper scripts for ArUco pose estimation.

## Files

### 1. `cli.py`
Main command-line interface for common operations.

**Usage:**
```bash
python cli.py [command] [options]
```

**Commands:**

#### Calibrate
Calibrate camera using checkerboard pattern:
```bash
python cli.py calibrate \
  --rows 9 \
  --cols 6 \
  --square-size 0.025 \
  --output calibration/
```

#### Track
Real-time marker tracking:
```bash
python cli.py track \
  --config config.yaml \
  --source 0 \
  --output tracking_data.json
```

#### Generate
Generate ArUco markers:
```bash
python cli.py generate \
  --ids 0 1 2 3 4 \
  --size 200 \
  --dict DICT_4X4_50 \
  --output markers/
```

#### Analyze
Analyze recorded tracking data:
```bash
python cli.py analyze \
  --input tracking_data.json \
  --plot \
  --report report.pdf
```

---

### 2. `visualize_aruco_extrinsics.py`
Visualize camera extrinsics and marker positions in 3D.

**Usage:**
```bash
python visualize_aruco_extrinsics.py \
  --config calibration.yaml \
  --markers markers.json
```

**What it does:**
- 3D visualization of camera frustum
- Marker positions and orientations
- Camera trajectory over time
- Interactive 3D viewer (matplotlib or Open3D)

**Features:**
- Camera coordinate system
- World coordinate system
- Marker coordinate frames
- Camera path visualization
- Export to various formats

---

## CLI Reference

### Global Options

- `--verbose, -v`: Verbose output
- `--quiet, -q`: Suppress output
- `--config, -c`: Config file path
- `--help, -h`: Show help message

---

### Calibrate Command

Calibrate camera intrinsics using checkerboard.

**Arguments:**
- `--rows`: Number of inner corners (height)
- `--cols`: Number of inner corners (width)
- `--square-size`: Physical size of squares (meters)
- `--output`: Output directory for calibration files
- `--source`: Camera source (default: 0)
- `--frames`: Number of frames to capture (default: 20)

**Example:**
```bash
python cli.py calibrate \
  --rows 9 --cols 6 \
  --square-size 0.025 \
  --frames 25 \
  --output calibration/
```

**Output files:**
- `camera_matrix.npy`: Camera intrinsic matrix
- `dist_coeffs.npy`: Distortion coefficients
- `calibration_report.txt`: Calibration statistics
- `frames/`: Captured calibration images

---

### Track Command

Real-time marker tracking with recording.

**Arguments:**
- `--config`: Configuration YAML file
- `--source`: Video source (camera index, file, or stream URL)
- `--output`: Output file for recorded data (JSON or CSV)
- `--show`: Show visualization (default: true)
- `--fps`: Target FPS (default: 30)
- `--duration`: Recording duration in seconds (default: unlimited)

**Example:**
```bash
# Track from camera 0
python cli.py track --config config.yaml --source 0

# Track from video file
python cli.py track --source video.mp4 --output poses.json

# Track for 30 seconds
python cli.py track --duration 30 --output data.csv
```

**Interactive controls:**
- `Space`: Pause/resume
- `R`: Start/stop recording
- `S`: Save current frame
- `Q` or `ESC`: Quit

---

### Generate Command

Generate ArUco markers for printing.

**Arguments:**
- `--ids`: Marker IDs to generate (space-separated)
- `--size`: Marker size in pixels (default: 200)
- `--dict`: ArUco dictionary (default: DICT_4X4_50)
- `--output`: Output directory (default: markers/)
- `--sheet`: Generate marker sheet (PDF)
- `--sheet-size`: Sheet size (A4, Letter, etc.)
- `--margin`: Margin in pixels (default: 20)

**Examples:**
```bash
# Generate single marker
python cli.py generate --ids 0 --size 200

# Generate multiple markers
python cli.py generate --ids 0 1 2 3 4 --size 150

# Generate marker sheet
python cli.py generate --ids 0 1 2 3 --sheet --sheet-size A4
```

**Supported dictionaries:**
- DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
- DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
- DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
- DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
- DICT_ARUCO_ORIGINAL

---

### Analyze Command

Analyze recorded tracking data.

**Arguments:**
- `--input`: Input data file (JSON or CSV)
- `--plot`: Generate plots
- `--report`: Generate PDF report
- `--metrics`: Compute tracking metrics
- `--export`: Export processed data

**Examples:**
```bash
# Basic analysis
python cli.py analyze --input tracking_data.json

# Generate plots
python cli.py analyze --input data.json --plot

# Full report
python cli.py analyze \
  --input data.json \
  --plot \
  --metrics \
  --report report.pdf
```

**Metrics computed:**
- Position accuracy
- Rotation accuracy
- Tracking stability
- Detection rate
- FPS statistics
- Reprojection error

**Plots generated:**
- Position over time (x, y, z)
- Orientation over time (roll, pitch, yaw)
- Trajectory visualization
- Error distributions
- FPS over time

---

## Visualization Tool

### visualize_aruco_extrinsics.py

Visualize 3D scene with cameras and markers.

**Usage:**
```bash
python visualize_aruco_extrinsics.py \
  --cameras camera_poses.json \
  --markers marker_positions.json \
  --output scene.png
```

**Arguments:**
- `--cameras`: Camera pose file (JSON)
- `--markers`: Marker position file (JSON)
- `--output`: Output image file
- `--interactive`: Interactive 3D viewer
- `--scale`: Scene scale factor
- `--grid`: Show grid

**Features:**
- 3D camera frustums
- Marker coordinate frames
- Camera trajectory
- Grid reference
- Axis labels
- Interactive rotation/zoom

**Input format (cameras):**
```json
{
  "cameras": [
    {
      "timestamp": 0.0,
      "position": [x, y, z],
      "orientation": [roll, pitch, yaw]
    }
  ]
}
```

**Input format (markers):**
```json
{
  "markers": [
    {
      "id": 0,
      "position": [x, y, z],
      "orientation": [roll, pitch, yaw]
    }
  ]
}
```

---

## Installation

All tools require the main package to be installed:

```bash
cd ..
pip install -e .
```

Or install dependencies manually:
```bash
pip install opencv-python numpy click pyyaml matplotlib
```

---

## Examples

### Complete workflow

1. **Calibrate camera:**
   ```bash
   python cli.py calibrate --rows 9 --cols 6 --output calibration/
   ```

2. **Generate markers:**
   ```bash
   python cli.py generate --ids 0 1 2 3 --size 200 --sheet
   ```

3. **Track markers:**
   ```bash
   python cli.py track --config config.yaml --output tracking.json
   ```

4. **Analyze results:**
   ```bash
   python cli.py analyze --input tracking.json --plot --report
   ```

5. **Visualize scene:**
   ```bash
   python visualize_aruco_extrinsics.py \
     --cameras tracking.json \
     --interactive
   ```

---

## Configuration

CLI can use config file to avoid repeating arguments:

**config.yaml:**
```yaml
calibration:
  rows: 9
  cols: 6
  square_size: 0.025
  output: calibration/

tracking:
  source: 0
  fps: 30
  show_visualization: true

generation:
  dictionary: DICT_4X4_50
  size: 200
  margin: 20

analysis:
  plot: true
  metrics: true
```

**Usage:**
```bash
python cli.py --config config.yaml calibrate
python cli.py --config config.yaml track
```

---

## Tips

### Calibration
- Capture images from various angles and distances
- Move checkerboard slowly to avoid motion blur
- Ensure entire checkerboard is visible
- Good lighting without glare

### Tracking
- Use `--show false` for headless recording
- CSV format is smaller than JSON
- Set `--fps` to match your needs (higher = more data)
- Use `--duration` to auto-stop recording

### Generation
- Print at 300+ DPI for best detection
- Use white border around markers
- Larger markers = easier detection from farther away
- Test detection before printing many markers

### Analysis
- Use `--metrics` to get quantitative results
- Generate plots to visualize tracking quality
- Check reprojection error distribution
- Compare multiple tracking sessions

---

## Troubleshooting

### "Command not found"
Ensure you're in the tools/ directory:
```bash
cd tools
python cli.py --help
```

### "Module not found"
Install package in development mode:
```bash
cd ..
pip install -e .
```

### "Camera not opening"
Try different camera index:
```bash
python cli.py track --source 1
```

Or specify video file:
```bash
python cli.py track --source path/to/video.mp4
```

### "Calibration failing"
- Check checkerboard dimensions
- Ensure good lighting
- Capture more frames (--frames 30)
- Move checkerboard to cover full image

---

## Extending

### Add custom command

Edit `cli.py`:

```python
@cli.command()
@click.option('--my-option', default='value')
def mycommand(my_option):
    """My custom command"""
    # Implementation
    pass
```

### Add custom visualization

Edit `visualize_aruco_extrinsics.py`:

```python
def my_visualization(data):
    # Custom visualization code
    pass
```

---

## API Usage

Import CLI functions in your own scripts:

```python
from tools.cli import calibrate_camera, track_markers

# Programmatic use
calibrate_camera(
    rows=9, cols=6,
    square_size=0.025,
    output_dir='calibration/'
)

track_markers(
    config='config.yaml',
    source=0,
    output='tracking.json'
)
```

---

## See Also

- [Examples](../examples/) - Example scripts
- [Documentation](../README_v2.md) - Main documentation
- [Quick Start](../QUICK_START.md) - Getting started guide
