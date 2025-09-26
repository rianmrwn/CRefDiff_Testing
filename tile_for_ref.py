#!/usr/bin/env python3
"""
Script to tile a single HR TIF file into smaller tiles.
(rasterio version)
"""

import os
import argparse
import numpy as np
import sys

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tile a HR TIF file into smaller tiles')
    parser.add_argument('--input', type=str, required=True, help='Path to input HR TIF file')
    parser.add_argument('--output_folder', type=str, default='/data_img/Ref/', help='Path to output folder for tiles')
    parser.add_argument('--tile_size', type=int, default=800, help='Tile size in pixels (default: 800)')
    return parser.parse_args()


def create_tiles(input_path, output_folder, tile_size):
    """
    Create tiles from a HR TIF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the input file
    try:
        dataset = rasterio.open(input_path)
    except Exception as e:
        print(f"Error: Could not open {input_path}: {e}")
        return False

    # Get image dimensions
    width = dataset.width
    height = dataset.height
    bands = dataset.count

    # Get transform and projection (CRS)
    transform = dataset.transform
    crs = dataset.crs

    # Calculate the number of tiles
    num_tiles_x = int(np.ceil(width / tile_size))
    num_tiles_y = int(np.ceil(height / tile_size))

    print(f"Image size: {width}x{height} pixels, {bands} bands")
    print(f"Creating {num_tiles_x}x{num_tiles_y} tiles of size {tile_size}x{tile_size}")

    # Get input filename without extension
    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    # Use source dtype for output tiles (fallback to float32 if unknown)
    out_dtype = dataset.dtypes[0] if dataset.dtypes and dataset.dtypes[0] else 'float32'

    # Create tiles
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            # Calculate pixel coordinates
            x_offset = i * tile_size
            y_offset = j * tile_size

            # Handle edge cases where tile would go beyond image boundaries
            x_size = min(tile_size, width - x_offset)
            y_size = min(tile_size, height - y_offset)

            # Skip if tile size is 0 in any dimension
            if x_size <= 0 or y_size <= 0:
                continue

            # Create output filename
            output_filename = f"tile_{i}_{j}.png"
            output_path = os.path.join(output_folder, output_filename)

            # Define window and its transform
            window = Window(x_offset, y_offset, x_size, y_size)
            new_transform = window_transform(window, transform)

            # Prepare output profile
            profile = dataset.profile.copy()
            profile.update({
                "width": x_size,
                "height": y_size,
                "count": bands,
                "dtype": out_dtype,
                "transform": new_transform,
                "crs": crs,
                "driver": "GTiff"
            })

            # Write the tile
            with rasterio.open(output_path, "w", **profile) as dst:
                for band in range(1, bands + 1):
                    data = dataset.read(band, window=window)
                    dst.write(data, band)

            print(f"Created tile: {output_filename}")

    # Close the input dataset
    dataset.close()
    return True


def main():
    """Main function"""
    args = parse_arguments()

    # Validate input file
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    # Validate tile size
    if args.tile_size <= 0:
        print(f"Error: Tile size must be positive, got {args.tile_size}")
        sys.exit(1)

    print(f"Processing file: {args.input}")
    print(f"Output folder: {args.output_folder}")
    print(f"Tile size: {args.tile_size}x{args.tile_size}")

    # Create tiles
    success = create_tiles(args.input, args.output_folder, args.tile_size)

    if success:
        print("Tiling completed successfully")
    else:
        print("Tiling failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
