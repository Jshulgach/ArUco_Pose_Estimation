"""
Simple wrapper for marker generation - demonstrates using src.utils.marker_generator.

For CLI usage, you can also use:
    python ../../tools/cli.py generate 0 1 2 --size 200
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.marker_generator import generate_aruco_marker, generate_marker_sheet
from src.utils.logger import setup_logger

logger = setup_logger("generate_markers_example")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ArUco markers')
    parser.add_argument('--id', type=int, default=0, help='Marker ID')
    parser.add_argument('--size', type=int, default=200, help='Marker size in pixels')
    parser.add_argument('--dict', type=str, default='DICT_4X4_50', help='ArUco dictionary')
    parser.add_argument('--output', type=str, default='markers/', help='Output directory')
    parser.add_argument('--sheet', action='store_true', help='Generate sheet of markers')
    parser.add_argument('--ids', nargs='+', type=int, help='Multiple marker IDs for sheet')
    
    args = parser.parse_args()
    
    if args.sheet and args.ids:
        logger.info(f"Generating marker sheet with IDs: {args.ids}")
        output_file = generate_marker_sheet(
            marker_ids=args.ids,
            dict_type=args.dict,
            marker_size_px=args.size,
            output_path=f"{args.output}/marker_sheet.png"
        )
        print(f"Marker sheet saved to: {output_file}")
    else:
        logger.info(f"Generating single marker ID {args.id}")
        output_file = generate_aruco_marker(
            marker_id=args.id,
            dict_type=args.dict,
            size_px=args.size,
            output_dir=args.output
        )
        print(f"Marker saved to: {output_file}")


if __name__ == "__main__":
    main()
