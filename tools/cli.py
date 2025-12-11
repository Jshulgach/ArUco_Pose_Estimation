"""
Command-line interface for ArUco Pose Estimation toolkit.
"""

import click
import cv2
import numpy as np
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import UnifiedArucoDetector
from src.utils.config import ProjectConfig
from src.utils.visualization import ArucoVisualizer
from src.utils.performance import PerformanceMonitor
from src.utils.logger import setup_logger
from src.utils.io import VideoHandler, PoseRecorder


@click.group()
@click.version_option(version='0.2.0')
def cli():
    """ArUco Pose Estimation Toolkit - Professional ArUco marker tracking and pose estimation"""
    pass


@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--visualize/--no-visualize', default=True, help='Show visualization during capture')
@click.option('--have-frames', is_flag=True, help='Skip capturing new frames')
def calibrate(config, visualize, have_frames):
    """Run camera calibration using checkerboard pattern."""
    logger = setup_logger("calibrate", level=10 if visualize else 20)
    
    try:
        # Load configuration
        if Path(config).exists():
            cfg = ProjectConfig.from_legacy_yaml(config)
        else:
            logger.warning(f"Config file not found: {config}, using defaults")
            cfg = ProjectConfig()
        
        cfg.validate()
        
        logger.info("Starting camera calibration...")
        logger.info(f"Checkerboard: {cfg.calibration.checkerboard_rows}x{cfg.calibration.checkerboard_cols}")
        logger.info(f"Square size: {cfg.calibration.square_size} m")
        
        # Import calibration utilities
        from utils import save_frames_single_camera, calibrate_single_camera
        
        if not have_frames:
            logger.info("Capturing calibration frames...")
            save_frames_single_camera(cfg.to_dict()['calibration'], visualize)
        
        logger.info("Computing calibration matrices...")
        _, mtx, dist = calibrate_single_camera(cfg.to_dict()['calibration'], visualize)
        
        # Save results
        np.save("camera_intrinsics.npy", mtx)
        np.save("camera_distortion.npy", dist)
        
        logger.info("✓ Calibration complete!")
        logger.info(f"  Saved: camera_intrinsics.npy, camera_distortion.npy")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        sys.exit(1)


@cli.command()
@click.option('--config', default='config.yaml', help='Path to config file')
@click.option('--source', default=0, help='Camera ID or video path')
@click.option('--calibration', help='Path to calibration matrix (.npy)')
@click.option('--distortion', help='Path to distortion coefficients (.npy)')
@click.option('--output', help='Output video file (optional)')
@click.option('--record', help='Record pose data to JSON file')
@click.option('--show-axes/--no-show-axes', default=True, help='Draw 3D axes on markers')
@click.option('--show-ids/--no-show-ids', default=True, help='Show marker IDs')
@click.option('--show-stats/--no-show-stats', default=True, help='Show performance stats')
@click.option('--marker-size', type=float, help='Physical marker size in meters')
def track(config, source, calibration, distortion, output, record, 
          show_axes, show_ids, show_stats, marker_size):
    """
    Run real-time ArUco marker tracking and pose estimation.
    
    Press 'q' to quit, 's' to save current frame, 'r' to reset statistics.
    """
    logger = setup_logger("track")
    
    try:
        # Load configuration
        if Path(config).exists():
            cfg = ProjectConfig.from_legacy_yaml(config)
        else:
            logger.warning(f"Config file not found: {config}, using defaults")
            cfg = ProjectConfig()
        
        # Override with command line arguments
        if marker_size:
            cfg.aruco.marker_size = marker_size
        
        # Load calibration
        K = None
        D = None
        if calibration or Path("camera_intrinsics.npy").exists():
            calib_path = calibration or "camera_intrinsics.npy"
            dist_path = distortion or "camera_distortion.npy"
            
            if Path(calib_path).exists() and Path(dist_path).exists():
                K = np.load(calib_path)
                D = np.load(dist_path)
                logger.info(f"Loaded calibration from {calib_path}")
            else:
                logger.warning("Calibration files not found, running without pose estimation")
        
        # Initialize components
        detector = UnifiedArucoDetector(cfg.aruco.dict_type)
        visualizer = ArucoVisualizer(K, D)
        monitor = PerformanceMonitor()
        
        # Initialize video handler
        try:
            source_val = int(source)
        except ValueError:
            source_val = source
        
        video = VideoHandler(source_val, cfg.camera.width, cfg.camera.height)
        logger.info(f"Opened video source: {source}")
        
        # Initialize recorder if requested
        recorder = None
        if record:
            recorder = PoseRecorder(record, save_interval=100)
        
        # Initialize video writer if output requested
        writer = None
        if output:
            props = video.get_properties()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, 30.0, 
                                    (props['width'], props['height']))
        
        logger.info("Starting tracking... (Press 'q' to quit)")
        frame_count = 0
        
        while True:
            monitor.start_frame()
            
            ret, frame = video.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            
            # Detect markers
            monitor.start_detection()
            corners, ids, rejected = detector.detect(frame)
            n_markers = len(ids) if ids is not None else 0
            monitor.end_detection(n_markers)
            
            # Draw markers
            frame = visualizer.draw_markers(frame, corners, ids, 
                                           draw_ids=show_ids,
                                           draw_axes=show_axes and K is not None,
                                           marker_size=cfg.aruco.marker_size)
            
            # Estimate pose if calibration available
            rvec, tvec = None, None
            if ids is not None and len(ids) > 0 and K is not None:
                monitor.start_pose_estimation()
                # Use first marker for simple pose estimation
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[0], cfg.aruco.marker_size, K, D
                )
                monitor.end_pose_estimation()
                
                # Draw pose info
                frame = visualizer.draw_pose_info(frame, rvec, tvec)
            
            # Record data if requested
            if recorder and rvec is not None:
                recorder.record_frame(time.time(), rvec, tvec, ids)
            
            # Show performance stats
            if show_stats:
                stats = monitor.get_stats()
                display_stats = {
                    'FPS': stats['fps'],
                    'Markers': n_markers,
                    'Frames': frame_count
                }
                frame = visualizer.draw_performance_stats(
                    frame, display_stats, position=(frame.shape[1]-200, 30)
                )
            
            # Save frame to video if requested
            if writer:
                writer.write(frame)
            
            # Display
            cv2.imshow('ArUco Tracker', frame)
            
            monitor.end_frame()
            frame_count += 1
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"frame_{frame_count:06d}.png"
                cv2.imwrite(save_path, frame)
                logger.info(f"Saved frame: {save_path}")
            elif key == ord('r'):
                monitor.reset()
                logger.info("Reset statistics")
        
        # Cleanup
        video.release()
        cv2.destroyAllWindows()
        
        if writer:
            writer.release()
            logger.info(f"Saved video: {output}")
        
        if recorder:
            recorder.save()
            stats = recorder.get_statistics()
            logger.info(f"Recorded {stats['total_frames']} frames")
        
        # Print final statistics
        logger.info("\n" + monitor.get_summary())
        logger.info("✓ Tracking complete!")
        
    except Exception as e:
        logger.error(f"Tracking failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.argument('marker_ids', nargs=-1, type=int, required=True)
@click.option('--type', '--dict-type', 'dict_type', default='DICT_5X5_100',
             help='ArUco dictionary type')
@click.option('--size', type=int, default=200, help='Marker size in pixels')
@click.option('--output', '-o', default='markers', help='Output directory')
@click.option('--sheet', is_flag=True, help='Generate a single printable sheet')
@click.option('--markers-per-row', type=int, default=4, help='Markers per row in sheet')
def generate(marker_ids, dict_type, size, output, sheet, markers_per_row):
    """
    Generate ArUco markers.
    
    Example: aruco-cli generate 0 1 2 3 4 5 --sheet --output markers/sheet.png
    """
    logger = setup_logger("generate")
    
    try:
        from src.utils.marker_generator import generate_aruco_marker, generate_marker_sheet
        
        if sheet:
            generate_marker_sheet(
                marker_ids=list(marker_ids),
                dict_type=dict_type,
                markers_per_row=markers_per_row,
                marker_size_px=size,
                output_path=output
            )
        else:
            for marker_id in marker_ids:
                generate_aruco_marker(
                    marker_id=marker_id,
                    dict_type=dict_type,
                    size_px=size,
                    output_dir=output
                )
        
        logger.info("✓ Marker generation complete!")
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


@cli.command()
@click.argument('video_path', type=click.Path(exists=True))
@click.option('--calibration', required=True, help='Path to calibration matrix (.npy)')
@click.option('--distortion', required=True, help='Path to distortion coefficients (.npy)')
@click.option('--dict-type', default='DICT_5X5_100', help='ArUco dictionary type')
@click.option('--marker-size', type=float, default=0.05, help='Marker size in meters')
@click.option('--output', help='Output analyzed video')
def analyze(video_path, calibration, distortion, dict_type, marker_size, output):
    """Analyze a recorded video and generate statistics."""
    logger = setup_logger("analyze")
    
    try:
        # Load calibration
        K = np.load(calibration)
        D = np.load(distortion)
        
        # Initialize components
        detector = UnifiedArucoDetector(dict_type)
        visualizer = ArucoVisualizer(K, D)
        monitor = PerformanceMonitor()
        
        video = VideoHandler(video_path)
        
        writer = None
        if output:
            props = video.get_properties()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output, fourcc, props['fps'], 
                                    (props['width'], props['height']))
        
        logger.info(f"Analyzing video: {video_path}")
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            corners, ids, _ = detector.detect(frame)
            n_markers = len(ids) if ids is not None else 0
            monitor.record_detection(n_markers)
            
            if writer:
                frame = visualizer.draw_markers(frame, corners, ids, 
                                               draw_axes=True, marker_size=marker_size)
                writer.write(frame)
        
        video.release()
        if writer:
            writer.release()
            logger.info(f"Saved analyzed video: {output}")
        
        logger.info("\n" + monitor.get_summary())
        logger.info("✓ Analysis complete!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
