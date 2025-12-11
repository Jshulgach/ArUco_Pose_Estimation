"""
Script to generate ArUco markers and marker sheets.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.detector import ARUCO_DICT
from src.utils.logger import setup_logger

logger = setup_logger("marker_generator")


def generate_aruco_marker(marker_id, dict_type="DICT_5X5_100", 
                          size_px=200, border_bits=1, output_dir="markers"):
    """
    Generate a single ArUco marker image.
    
    Args:
        marker_id: Marker ID to generate
        dict_type: ArUco dictionary type
        size_px: Marker size in pixels
        border_bits: Border width in bits
        output_dir: Output directory path
        
    Returns:
        Path to generated marker image
    """
    if dict_type not in ARUCO_DICT:
        raise ValueError(f"Invalid dict_type: {dict_type}. Must be one of {list(ARUCO_DICT.keys())}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    
    # Generate marker
    marker_img = np.zeros((size_px, size_px), dtype=np.uint8)
    marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, size_px, marker_img, border_bits)
    
    # Save marker
    output_file = output_path / f"{dict_type}_id_{marker_id}.png"
    cv2.imwrite(str(output_file), marker_img)
    
    logger.info(f"Generated marker ID {marker_id} ({dict_type}) -> {output_file}")
    return output_file


def generate_marker_sheet(marker_ids, dict_type="DICT_5X5_100", 
                         markers_per_row=4, marker_size_px=200, 
                         spacing_px=20, border_bits=1,
                         output_path="markers/marker_sheet.png"):
    """
    Generate a printable sheet with multiple markers.
    
    Args:
        marker_ids: List of marker IDs to include
        dict_type: ArUco dictionary type
        markers_per_row: Number of markers per row
        marker_size_px: Individual marker size in pixels
        spacing_px: Spacing between markers
        border_bits: Border width in bits
        output_path: Output file path
        
    Returns:
        Path to generated sheet
    """
    if dict_type not in ARUCO_DICT:
        raise ValueError(f"Invalid dict_type: {dict_type}")
    
    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    
    # Calculate sheet dimensions
    n_markers = len(marker_ids)
    n_rows = (n_markers + markers_per_row - 1) // markers_per_row
    
    sheet_width = markers_per_row * marker_size_px + (markers_per_row + 1) * spacing_px
    sheet_height = n_rows * marker_size_px + (n_rows + 1) * spacing_px
    
    # Create white sheet
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    # Generate and place markers
    for i, marker_id in enumerate(marker_ids):
        row = i // markers_per_row
        col = i % markers_per_row
        
        # Generate marker
        marker_img = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size_px, 
                                         marker_img, border_bits)
        
        # Calculate position
        y_pos = spacing_px + row * (marker_size_px + spacing_px)
        x_pos = spacing_px + col * (marker_size_px + spacing_px)
        
        # Place marker on sheet
        sheet[y_pos:y_pos+marker_size_px, x_pos:x_pos+marker_size_px] = marker_img
        
        # Add ID label below marker
        label = f"ID: {marker_id}"
        label_pos = (x_pos + 5, y_pos + marker_size_px + 15)
        cv2.putText(sheet, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, 0, 1, cv2.LINE_AA)
    
    # Save sheet
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), sheet)
    
    logger.info(f"Generated marker sheet with {n_markers} markers -> {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers and marker sheets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single marker
  python generate_markers.py --ids 0 --type DICT_5X5_100 --output markers/
  
  # Generate multiple markers
  python generate_markers.py --ids 0 1 2 3 4 5 --output markers/
  
  # Generate printable sheet
  python generate_markers.py --ids 0 1 2 3 4 5 --sheet --output markers/sheet.png
        """
    )
    
    parser.add_argument('--ids', type=int, nargs='+', required=True,
                       help='Marker IDs to generate')
    parser.add_argument('--type', '--dict-type', dest='dict_type',
                       default='DICT_5X5_100',
                       choices=list(ARUCO_DICT.keys()),
                       help='ArUco dictionary type')
    parser.add_argument('--size', type=int, default=200,
                       help='Marker size in pixels (default: 200)')
    parser.add_argument('--border', type=int, default=1,
                       help='Border width in bits (default: 1)')
    parser.add_argument('--output', '-o', default='markers',
                       help='Output directory or file path')
    parser.add_argument('--sheet', action='store_true',
                       help='Generate a single sheet with all markers')
    parser.add_argument('--markers-per-row', type=int, default=4,
                       help='Markers per row in sheet (default: 4)')
    parser.add_argument('--spacing', type=int, default=20,
                       help='Spacing between markers in sheet (default: 20)')
    
    args = parser.parse_args()
    
    try:
        if args.sheet:
            # Generate marker sheet
            generate_marker_sheet(
                marker_ids=args.ids,
                dict_type=args.dict_type,
                markers_per_row=args.markers_per_row,
                marker_size_px=args.size,
                spacing_px=args.spacing,
                border_bits=args.border,
                output_path=args.output
            )
        else:
            # Generate individual markers
            for marker_id in args.ids:
                generate_aruco_marker(
                    marker_id=marker_id,
                    dict_type=args.dict_type,
                    size_px=args.size,
                    border_bits=args.border,
                    output_dir=args.output
                )
        
        logger.info("✓ Marker generation complete!")
        
    except Exception as e:
        logger.error(f"Error generating markers: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
