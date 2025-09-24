# combined_sen2_processing.py

import os
import math
import time
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin
import argparse
from tqdm import tqdm
from io import BytesIO
import pyproj

# Import the downloader classes from both scripts
from sen2tobing import BingMapDownloader
from sen2togooglesat import GoogleSatelliteDownloader

def process_both_sources(args):
    """
    Process both Bing Maps and Google Satellite imagery using the same parameters
    """
    # Initialize both downloaders
    bing_downloader = BingMapDownloader()
    google_downloader = GoogleSatelliteDownloader()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.bing_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.google_output), exist_ok=True)

    # Process with reference TIF if provided
    if args.input_tif_reference:
        print("\nProcessing Bing Maps imagery...")
        bing_downloader.download_from_reference_tif(
            args.input_tif_reference,
            args.bing_output,
            args.resolution
        )
        
        print("\nProcessing Google Satellite imagery...")
        google_downloader.download_from_reference_tif(
            args.input_tif_reference,
            args.google_output,
            args.resolution
        )
    
    # Process with explicit coordinates if provided
    elif all(v is not None for v in [args.center_lat, args.center_lon, args.width, args.height]):
        print("\nProcessing Bing Maps imagery...")
        bing_downloader.download_area(
            args.center_lat,
            args.center_lon,
            args.width,
            args.height,
            args.resolution,
            args.bing_output
        )
        
        print("\nProcessing Google Satellite imagery...")
        google_downloader.download_area(
            args.center_lat,
            args.center_lon,
            args.width,
            args.height,
            args.resolution,
            args.google_output
        )
    
    else:
        raise ValueError("Either input_tif_reference or all coordinate parameters must be provided")

def main():
    parser = argparse.ArgumentParser(description='Download high-resolution satellite imagery from both Bing Maps and Google Satellite')
    
    # Input parameters
    parser.add_argument('--center_lat', type=float, help='Center latitude')
    parser.add_argument('--center_lon', type=float, help='Center longitude')
    parser.add_argument('--width', type=float, help='Width in meters')
    parser.add_argument('--height', type=float, help='Height in meters')
    parser.add_argument('--resolution', type=float, default=0.75, 
                       help='Resolution in meters per pixel (default: 0.75)')
    
    # Output paths
    parser.add_argument('--bing_output', type=str, 
                       default='/data_img/data_prep/Ref.tif',
                       help='Output path for Bing Maps imagery')
    parser.add_argument('--google_output', type=str,
                       default='/data_img/data_prep/HR.tif',
                       help='Output path for Google Satellite imagery')
    
    # Reference TIF
    parser.add_argument('--input_tif_reference', type=str,
                       help='Reference GeoTIFF for area and resolution')

    args = parser.parse_args()

    try:
        process_both_sources(args)
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()

# python combined_sen2_processing.py \
#     --input_tif_reference /path/to/reference.tif \
#     --bing_output /path/to/output/bing.tif \
#     --google_output /path/to/output/google.tif \
#     --resolution 0.75
