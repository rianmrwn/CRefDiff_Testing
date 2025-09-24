# combined_tile_processing.py

import os
import argparse
import numpy as np
from osgeo import gdal
import sys
from enum import Enum

class ImageType(Enum):
    HR = "hr"
    REF = "ref"
    SENTINEL = "sentinel"

class TileProcessor:
    def __init__(self, image_type, input_path, output_folder, tile_size):
        self.image_type = image_type
        self.input_path = input_path
        self.output_folder = output_folder
        self.tile_size = tile_size
        
        # Set default tile sizes based on image type
        self.default_sizes = {
            ImageType.HR: 800,
            ImageType.REF: 800,
            ImageType.SENTINEL: 48
        }
        
        if tile_size is None:
            self.tile_size = self.default_sizes[image_type]

    def create_tiles(self):
        """Create tiles from input TIF file"""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Open the input file
        dataset = gdal.Open(self.input_path)
        if dataset is None:
            print(f"Error: Could not open {self.input_path}")
            return False
        
        try:
            # Get image dimensions
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            bands = dataset.RasterCount
            
            # Get geotransform and projection
            geotransform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            
            # Calculate the number of tiles
            num_tiles_x = int(np.ceil(width / self.tile_size))
            num_tiles_y = int(np.ceil(height / self.tile_size))
            
            print(f"\nProcessing {self.image_type.value.upper()} image:")
            print(f"Image size: {width}x{height} pixels, {bands} bands")
            print(f"Creating {num_tiles_x}x{num_tiles_y} tiles of size {self.tile_size}x{self.tile_size}")
            
            # Get input filename without extension
            base_filename = os.path.splitext(os.path.basename(self.input_path))[0]
            
            # Create tiles
            for i in range(num_tiles_x):
                for j in range(num_tiles_y):
                    self._process_single_tile(dataset, base_filename, i, j, width, height, bands, geotransform, projection)
            
            return True
            
        finally:
            # Close the input dataset
            dataset = None

    def _process_single_tile(self, dataset, base_filename, i, j, width, height, bands, geotransform, projection):
        """Process a single tile"""
        # Calculate pixel coordinates
        x_offset = i * self.tile_size
        y_offset = j * self.tile_size
        
        # Handle edge cases where tile would go beyond image boundaries
        x_size = min(self.tile_size, width - x_offset)
        y_size = min(self.tile_size, height - y_offset)
        
        # Skip if tile size is 0 in any dimension
        if x_size <= 0 or y_size <= 0:
            return
        
        # Create output filename
        output_filename = f"{base_filename}_{self.image_type.value}_tile_{i}_{j}.tif"
        output_path = os.path.join(self.output_folder, output_filename)
        
        # Calculate new geotransform for this tile
        new_geotransform = list(geotransform)
        new_geotransform[0] = geotransform[0] + x_offset * geotransform[1]
        new_geotransform[3] = geotransform[3] + y_offset * geotransform[5]
        
        # Create the output file
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            output_path, 
            x_size, 
            y_size, 
            bands, 
            gdal.GDT_Float32
        )
        
        # Set the geotransform and projection
        output_dataset.SetGeoTransform(new_geotransform)
        output_dataset.SetProjection(projection)
        
        # Copy the data for each band
        for band in range(1, bands + 1):
            data = dataset.GetRasterBand(band).ReadAsArray(x_offset, y_offset, x_size, y_size)
            output_dataset.GetRasterBand(band).WriteArray(data)
        
        # Close the dataset
        output_dataset = None
        
        print(f"Created tile: {output_filename}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process and tile multiple types of satellite imagery')
    
    # Input files
    parser.add_argument('--hr_input', type=str, help='Path to input HR TIF file')
    parser.add_argument('--ref_input', type=str, help='Path to input Reference TIF file')
    parser.add_argument('--sentinel_input', type=str, help='Path to input Sentinel-2 TIF file')
    
    # Output folders
    parser.add_argument('--output_base', type=str, required=True, help='Base output directory for all tiles')
    
    # Optional tile sizes
    parser.add_argument('--hr_tile_size', type=int, help='Tile size for HR image (default: 800)')
    parser.add_argument('--ref_tile_size', type=int, help='Tile size for Reference image (default: 800)')
    parser.add_argument('--sentinel_tile_size', type=int, help='Tile size for Sentinel image (default: 48)')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create base output directory
    os.makedirs(args.output_base, exist_ok=True)
    
    # Define processing configurations
    configs = [
        (ImageType.HR, args.hr_input, "hr_tiles", args.hr_tile_size),
        (ImageType.REF, args.ref_input, "ref_tiles", args.ref_tile_size),
        (ImageType.SENTINEL, args.sentinel_input, "sentinel_tiles", args.sentinel_tile_size)
    ]
    
    # Process each image type
    for img_type, input_path, subfolder, tile_size in configs:
        if input_path:
            if not os.path.isfile(input_path):
                print(f"Error: Input file {input_path} does not exist")
                continue
                
            output_folder = os.path.join(args.output_base, subfolder)
            processor = TileProcessor(img_type, input_path, output_folder, tile_size)
            
            print(f"\nProcessing {img_type.value.upper()} image:")
            print(f"Input: {input_path}")
            print(f"Output: {output_folder}")
            print(f"Tile size: {processor.tile_size}x{processor.tile_size}")
            
            success = processor.create_tiles()
            
            if success:
                print(f"{img_type.value.upper()} tiling completed successfully")
            else:
                print(f"{img_type.value.upper()} tiling failed")

if __name__ == "__main__":
    main()

# python combined_tile_processing.py \
#     --hr_input /path/to/hr.tif \
#     --ref_input /path/to/ref.tif \
#     --sentinel_input /path/to/sentinel.tif \
#     --output_base /path/to/output/directory
