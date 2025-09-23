import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from PIL import Image
import glob
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Stitch super-resolution tiles into georeferenced image")
    
    parser.add_argument("--input_dir", required=True, type=str, 
                       help="Directory containing super-resolution tiles")
    parser.add_argument("--output_path", required=True, type=str,
                       help="Output path for stitched georeferenced image")
    parser.add_argument("--tile_info", type=str,
                       help="JSON file containing tile georeference information")
    parser.add_argument("--original_raster", type=str,
                       help="Original raster file to copy georeference from")
    parser.add_argument("--tile_size", type=int, default=800,
                       help="Size of each tile in pixels")
    parser.add_argument("--overlap", type=int, default=0,
                       help="Overlap between tiles in pixels")
    parser.add_argument("--scale_factor", type=float, default=16.67,
                       help="Super-resolution scale factor (800/48)")
    parser.add_argument("--blend_mode", type=str, default="feather", 
                       choices=["simple", "feather", "linear"],
                       help="Blending mode for overlapping areas")
    
    return parser.parse_args()


class TileStitcher:
    def __init__(self, tile_size: int = 800, overlap: int = 0, scale_factor: float = 16.67):
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        
    def load_tile_info(self, tile_info_path: str) -> dict:
        """Load tile georeference information from JSON file"""
        with open(tile_info_path, 'r') as f:
            return json.load(f)
    
    def extract_tile_coordinates(self, filename: str) -> Tuple[int, int]:
        """Extract tile coordinates from filename (assuming format: tile_row_col.png)"""
        # Example: tile_0_1.png -> (0, 1)
        parts = Path(filename).stem.split('_')
        if len(parts) >= 3 and parts[0] == 'tile':
            return int(parts[1]), int(parts[2])
        
        # Alternative: extract from numeric parts
        numbers = [int(s) for s in parts if s.isdigit()]
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
        
        raise ValueError(f"Cannot extract coordinates from filename: {filename}")
    
    def create_feather_mask(self, tile_shape: Tuple[int, int], overlap: int) -> np.ndarray:
        """Create feathering mask for smooth blending"""
        h, w = tile_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        if overlap > 0:
            # Create feathering at edges
            feather_size = min(overlap, min(h, w) // 4)
            
            # Top edge
            for i in range(feather_size):
                mask[i, :] *= i / feather_size
            
            # Bottom edge
            for i in range(feather_size):
                mask[h-1-i, :] *= i / feather_size
            
            # Left edge
            for j in range(feather_size):
                mask[:, j] *= j / feather_size
            
            # Right edge
            for j in range(feather_size):
                mask[:, w-1-j] *= j / feather_size
        
        return mask
    
    def stitch_tiles_simple(self, tiles_dir: str, output_path: str, 
                           tile_info: Optional[dict] = None) -> None:
        """Simple tile stitching without georeference"""
        
        # Find all tile images
        tile_files = glob.glob(os.path.join(tiles_dir, "*.png"))
        tile_files.extend(glob.glob(os.path.join(tiles_dir, "*.jpg")))
        
        if not tile_files:
            raise ValueError(f"No tile images found in {tiles_dir}")
        
        # Extract tile coordinates and determine grid size
        tile_coords = {}
        max_row, max_col = 0, 0
        
        for tile_file in tile_files:
            try:
                row, col = self.extract_tile_coordinates(os.path.basename(tile_file))
                tile_coords[(row, col)] = tile_file
                max_row = max(max_row, row)
                max_col = max(max_col, col)
            except ValueError as e:
                print(f"Skipping file {tile_file}: {e}")
                continue
        
        print(f"Found {len(tile_coords)} tiles, grid size: {max_row+1} x {max_col+1}")
        
        # Load first tile to get dimensions
        first_tile = Image.open(list(tile_coords.values())[0])
        tile_h, tile_w = first_tile.size[1], first_tile.size[0]
        channels = len(first_tile.getbands())
        
        # Calculate output dimensions
        effective_tile_size = self.tile_size - self.overlap
        output_h = (max_row + 1) * effective_tile_size + self.overlap
        output_w = (max_col + 1) * effective_tile_size + self.overlap
        
        # Initialize output array
        output_image = np.zeros((output_h, output_w, channels), dtype=np.float32)
        weight_map = np.zeros((output_h, output_w), dtype=np.float32)
        
        # Create feather mask
        feather_mask = self.create_feather_mask((tile_h, tile_w), self.overlap)
        
        # Stitch tiles
        for (row, col), tile_file in tile_coords.items():
            print(f"Processing tile ({row}, {col}): {os.path.basename(tile_file)}")
            
            # Load tile
            tile_img = np.array(Image.open(tile_file)).astype(np.float32)
            if len(tile_img.shape) == 2:  # Grayscale
                tile_img = tile_img[:, :, np.newaxis]
            
            # Calculate position in output image
            start_row = row * effective_tile_size
            start_col = col * effective_tile_size
            end_row = start_row + tile_h
            end_col = start_col + tile_w
            
            # Ensure we don't exceed output bounds
            end_row = min(end_row, output_h)
            end_col = min(end_col, output_w)
            
            # Get the actual tile region that fits
            tile_region_h = end_row - start_row
            tile_region_w = end_col - start_col
            
            # Apply feather mask
            current_mask = feather_mask[:tile_region_h, :tile_region_w]
            
            # Blend into output
            for c in range(channels):
                output_image[start_row:end_row, start_col:end_col, c] += \
                    tile_img[:tile_region_h, :tile_region_w, c] * current_mask
            
            weight_map[start_row:end_row, start_col:end_col] += current_mask
        
        # Normalize by weights
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        for c in range(channels):
            output_image[:, :, c] /= weight_map
        
        # Convert back to uint8 and save
        output_image = np.clip(output_image, 0, 255).astype(np.uint8)
        if channels == 1:
            output_image = output_image[:, :, 0]
        
        Image.fromarray(output_image).save(output_path)
        print(f"Stitched image saved to: {output_path}")
    
    def stitch_tiles_georeferenced(self, tiles_dir: str, output_path: str,
                                  tile_info: dict, original_raster: Optional[str] = None) -> None:
        """Stitch tiles with georeference preservation"""
        
        # Find all tile images
        tile_files = glob.glob(os.path.join(tiles_dir, "*.png"))
        tile_files.extend(glob.glob(os.path.join(tiles_dir, "*.jpg")))
        
        if not tile_files:
            raise ValueError(f"No tile images found in {tiles_dir}")
        
        # Create temporary georeferenced tiles
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_geotiles")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_files = []
        
        try:
            # Process each tile
            for tile_file in tile_files:
                tile_name = os.path.basename(tile_file)
                
                if tile_name not in tile_info:
                    print(f"No georeference info for {tile_name}, skipping...")
                    continue
                
                # Load tile image
                tile_img = np.array(Image.open(tile_file))
                if len(tile_img.shape) == 2:  # Grayscale
                    tile_img = tile_img[:, :, np.newaxis]
                
                # Get georeference info
                tile_bounds = tile_info[tile_name]['bounds']  # [minx, miny, maxx, maxy]
                tile_crs = tile_info[tile_name].get('crs', 'EPSG:4326')
                
                # Calculate transform for super-resolved tile
                sr_height, sr_width = tile_img.shape[:2]
                transform = from_bounds(
                    tile_bounds[0], tile_bounds[1], 
                    tile_bounds[2], tile_bounds[3],
                    sr_width, sr_height
                )
                
                # Create temporary georeferenced file
                temp_file = os.path.join(temp_dir, f"geo_{tile_name.replace('.png', '.tif')}")
                
                with rasterio.open(
                    temp_file, 'w',
                    driver='GTiff',
                    height=sr_height,
                    width=sr_width,
                    count=tile_img.shape[2] if len(tile_img.shape) == 3 else 1,
                    dtype=tile_img.dtype,
                    crs=tile_crs,
                    transform=transform,
                    compress='lzw'
                ) as dst:
                    if len(tile_img.shape) == 3:
                        for i in range(tile_img.shape[2]):
                            dst.write(tile_img[:, :, i], i + 1)
                    else:
                        dst.write(tile_img, 1)
                
                temp_files.append(temp_file)
                print(f"Created georeferenced tile: {temp_file}")
            
            # Merge all georeferenced tiles
            print("Merging georeferenced tiles...")
            
            # Open all temp files
            src_files = [rasterio.open(f) for f in temp_files]
            
            # Merge
            mosaic, out_trans = merge(src_files, method='first')
            
            # Get metadata from first file
            out_meta = src_files[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw"
            })
            
            # Write merged result
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)
            
            # Close source files
            for src in src_files:
                src.close()
            
            print(f"Georeferenced stitched image saved to: {output_path}")
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    def copy_georeference_from_original(self, original_raster: str, tiles_dir: str, 
                                      output_path: str) -> None:
        """Copy georeference from original raster and scale appropriately"""
        
        with rasterio.open(original_raster) as src:
            # Get original metadata
            original_transform = src.transform
            original_crs = src.crs
            original_bounds = src.bounds
            original_height, original_width = src.height, src.width
        
        # First stitch without georeference
        temp_output = output_path.replace('.tif', '_temp.png')
        self.stitch_tiles_simple(tiles_dir, temp_output)
        
        # Load stitched image
        stitched_img = np.array(Image.open(temp_output))
        if len(stitched_img.shape) == 2:
            stitched_img = stitched_img[:, :, np.newaxis]
        
        sr_height, sr_width = stitched_img.shape[:2]
        
        # Calculate new transform (scaled)
        # The pixel size should be smaller by the scale factor
        new_pixel_size_x = original_transform.a / self.scale_factor
        new_pixel_size_y = original_transform.e / self.scale_factor
        
        new_transform = rasterio.transform.Affine(
            new_pixel_size_x, 0.0, original_bounds.left,
            0.0, new_pixel_size_y, original_bounds.top
        )
        
        # Write georeferenced result
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=sr_height,
            width=sr_width,
            count=stitched_img.shape[2] if len(stitched_img.shape) == 3 else 1,
            dtype=stitched_img.dtype,
            crs=original_crs,
            transform=new_transform,
            compress='lzw'
        ) as dst:
            if len(stitched_img.shape) == 3:
                for i in range(stitched_img.shape[2]):
                    dst.write(stitched_img[:, :, i], i + 1)
            else:
                dst.write(stitched_img, 1)
        
        # Clean up temp file
        os.remove(temp_output)
        print(f"Georeferenced image saved to: {output_path}")


def main():
    args = parse_args()
    
    stitcher = TileStitcher(
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale_factor=args.scale_factor
    )
    
    # Determine which stitching method to use
    if args.tile_info and os.path.exists(args.tile_info):
        print("Using tile georeference information...")
        tile_info = stitcher.load_tile_info(args.tile_info)
        stitcher.stitch_tiles_georeferenced(args.input_dir, args.output_path, tile_info)
        
    elif args.original_raster and os.path.exists(args.original_raster):
        print("Copying georeference from original raster...")
        stitcher.copy_georeference_from_original(args.original_raster, args.input_dir, args.output_path)
        
    else:
        print("No georeference information provided, creating simple stitched image...")
        stitcher.stitch_tiles_simple(args.input_dir, args.output_path)
    
    print("Stitching completed!")


if __name__ == "__main__":
    main()
