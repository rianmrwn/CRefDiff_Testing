#!/usr/bin/env python3
"""
Script to tile a single Sentinel-2 TIF file into smaller tiles.
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import sys


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Tile a Sentinel-2 TIF file into smaller tiles')
    parser.add_argument('--input', type=str, required=True, help='Path to input Sentinel-2 TIF file')
    parser.add_argument('--output_folder', type=str, default='/data_img/temp_lr/', help='Path to output folder for tiles')
    parser.add_argument('--tile_size', type=int, default=48, help='Tile size in pixels (default: 256)')
    
    return parser.parse_args()


def create_tiles(input_path, output_folder, tile_size):
    """
    Create tiles from a Sentinel-2 TIF file
    
    Args:
        input_path (str): Path to the input TIF file
        output_folder (str): Path to the output folder
        tile_size (int): Size of tiles in pixels
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the input file
    with rasterio.open(input_path) as src:
        # Get image dimensions
        width = src.width
        height = src.height
        bands = src.count
        
        # Get metadata
        profile = src.profile
        
        # Calculate the number of tiles
        num_tiles_x = int(np.ceil(width / tile_size))
        num_tiles_y = int(np.ceil(height / tile_size))
        
        print(f"Image size: {width}x{height} pixels, {bands} bands")
        print(f"Creating {num_tiles_x}x{num_tiles_y} tiles of size {tile_size}x{tile_size}")
        
        # Get input filename without extension
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        
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
                output_filename = f"{base_filename}_tile_{i}_{j}.tif"
                output_path = os.path.join(output_folder, output_filename)
                
                # Create a window for reading the data
                window = Window(x_offset, y_offset, x_size, y_size)
                
                # Update the transform for the tile
                transform = rasterio.windows.transform(window, src.transform)
                
                # Update profile for the output file
                profile_output = profile.copy()
                profile_output.update({
                    'height': y_size,
                    'width': x_size,
                    'transform': transform
                })
                
                # Read the data for all bands
                data = src.read(window=window)
                
                # Write the tile
                with rasterio.open(output_path, 'w', **profile_output) as dst:
                    dst.write(data)
                
                print(f"Created tile: {output_filename}")
    
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
