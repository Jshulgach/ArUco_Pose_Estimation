"""
Verification script to test all new components.
Run this to verify the installation and new features work correctly.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all new modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.core.detector import UnifiedArucoDetector, ARUCO_DICT
        from src.utils.config import ProjectConfig, CameraConfig, CalibrationConfig, ArucoConfig
        from src.utils.visualization import ArucoVisualizer
        from src.utils.performance import PerformanceMonitor
        from src.utils.logger import setup_logger
        from src.utils.io import VideoHandler, PoseRecorder
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_detector():
    """Test UnifiedArucoDetector"""
    print("\nTesting UnifiedArucoDetector...")
    
    try:
        from src.core.detector import UnifiedArucoDetector
        import cv2
        
        detector = UnifiedArucoDetector("DICT_5X5_100")
        
        # Create a test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Try detection (should return empty)
        corners, ids, rejected = detector.detect(test_image)
        
        print(f"  Detector initialized: {detector}")
        print(f"  Detection result: corners={type(corners)}, ids={type(ids)}")
        print("✓ Detector works correctly")
        return True
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        return False


def test_config():
    """Test configuration management"""
    print("\nTesting configuration management...")
    
    try:
        from src.utils.config import ProjectConfig, CameraConfig, CalibrationConfig, ArucoConfig
        
        # Create config
        config = ProjectConfig(
            camera=CameraConfig(device_id=0, width=1920, height=1080),
            calibration=CalibrationConfig(checkerboard_rows=5, checkerboard_cols=7),
            aruco=ArucoConfig(dict_type="DICT_5X5_100", marker_size=0.05)
        )
        
        # Validate
        config.validate()
        
        # Save and load
        test_config_path = "test_config.yaml"
        config.to_yaml(test_config_path)
        loaded_config = ProjectConfig.from_yaml(test_config_path)
        
        # Cleanup
        Path(test_config_path).unlink()
        
        print(f"  Config created and validated")
        print(f"  Camera: {loaded_config.camera.width}x{loaded_config.camera.height}")
        print(f"  Marker size: {loaded_config.aruco.marker_size}")
        print("✓ Configuration management works correctly")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_visualization():
    """Test visualization tools"""
    print("\nTesting visualization...")
    
    try:
        from src.utils.visualization import ArucoVisualizer
        import cv2
        
        # Create test data
        K = np.eye(3) * 1000
        D = np.zeros(5)
        
        viz = ArucoVisualizer(K, D)
        
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test drawing (with no markers)
        result = viz.draw_markers(test_image, [], None)
        
        # Test performance stats
        stats = {'FPS': 30.5, 'Markers': 5}
        result = viz.draw_performance_stats(test_image, stats)
        
        print(f"  Visualizer created")
        print(f"  Drawing methods work")
        print("✓ Visualization works correctly")
        return True
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
        return False


def test_performance():
    """Test performance monitoring"""
    print("\nTesting performance monitoring...")
    
    try:
        from src.utils.performance import PerformanceMonitor
        import time
        
        monitor = PerformanceMonitor(window_size=10)
        
        # Simulate some frames
        for i in range(5):
            monitor.start_frame()
            time.sleep(0.01)  # Simulate work
            monitor.record_detection(num_markers=i)
            monitor.end_frame()
        
        # Get stats
        stats = monitor.get_stats()
        summary = monitor.get_summary()
        
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Avg markers: {stats['avg_markers']:.1f}")
        print("✓ Performance monitoring works correctly")
        return True
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False


def test_logger():
    """Test logging system"""
    print("\nTesting logging system...")
    
    try:
        from src.utils.logger import setup_logger
        import tempfile
        
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_path = f.name
        
        # Setup logger
        logger = setup_logger("test_logger", log_file=log_path, level=20)
        
        # Test logging
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Check log file
        with open(log_path, 'r') as f:
            log_content = f.read()
        
        # Cleanup
        Path(log_path).unlink()
        
        print(f"  Logger created")
        print(f"  Log file written: {len(log_content)} bytes")
        print("✓ Logging system works correctly")
        return True
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        return False


def test_io():
    """Test I/O utilities"""
    print("\nTesting I/O utilities...")
    
    try:
        from src.utils.io import PoseRecorder
        import time
        import tempfile
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            record_path = f.name
        
        # Create recorder
        recorder = PoseRecorder(record_path, save_interval=0)
        
        # Record some frames
        for i in range(3):
            rvec = np.random.randn(3, 1)
            tvec = np.random.randn(3, 1)
            ids = np.array([0, 1])
            recorder.record_frame(time.time(), rvec, tvec, ids, reprojection_error=0.5)
        
        # Save
        recorder.save()
        
        # Get statistics
        stats = recorder.get_statistics()
        
        # Cleanup
        Path(record_path).unlink()
        
        print(f"  Recorded {stats['total_frames']} frames")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print("✓ I/O utilities work correctly")
        return True
    except Exception as e:
        print(f"✗ I/O test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("ArUco Pose Estimation v2.0 - Verification Tests")
    print("="*60)
    
    tests = [
        test_imports,
        test_detector,
        test_config,
        test_visualization,
        test_performance,
        test_logger,
        test_io,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is working correctly.")
        print("\nNext steps:")
        print("  1. Read QUICK_START.md for usage examples")
        print("  2. Try: python examples/basic_detection.py")
        print("  3. Generate markers: python scripts/generate_markers.py --ids 0 1 2 --sheet")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("Check the error messages above for details.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
