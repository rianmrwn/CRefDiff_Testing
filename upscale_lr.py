import os
import glob
import pandas as pd
from osgeo import gdal, osr
import numpy as np
from pathlib import Path
import json
from datetime import datetime

class BatchGSDProcessor:
    """
    Batch processor for calculating and verifying GSD across multiple GeoTIFF files
    """
    
    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "processed"
        self.results = []
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def get_file_list(self, pattern="*.tif"):
        """
        Get list of GeoTIFF files in the input folder
        """
        file_patterns = [pattern, "*.tiff", "*.TIF", "*.TIFF"]
        files = []
        
        for pattern in file_patterns:
            files.extend(list(self.input_folder.glob(pattern)))
            
        return sorted(files)
    
    def calculate_gsd_from_file(self, file_path):
        """
        Calculate GSD from a single GeoTIFF file
        """
        try:
            dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
            if dataset is None:
                return None, f"Could not open file: {file_path}"
            
            # Get geotransform and basic info
            geotransform = dataset.GetGeoTransform()
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            bands = dataset.RasterCount
            
            # Calculate GSD
            gsd_x = abs(geotransform[1])
            gsd_y = abs(geotransform[5])
            
            # Get coordinate system info
            projection = dataset.GetProjection()
            srs = osr.SpatialReference()
            srs.ImportFromWkt(projection)
            
            # Get geographic extent
            min_x = geotransform[0]
            max_y = geotransform[3]
            max_x = min_x + (width * geotransform[1])
            min_y = max_y + (height * geotransform[5])
            
            file_info = {
                'filename': file_path.name,
                'filepath': str(file_path),
                'width': width,
                'height': height,
                'bands': bands,
                'gsd_x': gsd_x,
                'gsd_y': gsd_y,
                'gsd_avg': (gsd_x + gsd_y) / 2,
                'geographic_width': width * gsd_x,
                'geographic_height': height * gsd_y,
                'total_area_sqm': (width * gsd_x) * (height * gsd_y),
                'extent': {
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y
                },
                'projection': srs.GetAttrValue('AUTHORITY', 1) if srs.GetAttrValue('AUTHORITY') else 'Unknown',
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
            dataset = None
            return file_info, None
            
        except Exception as e:
            return None, f"Error processing {file_path}: {str(e)}"
    
    def batch_upscale_with_gsd_calculation(self, target_size=(800, 800), resample_method='cubic'):
        """
        Batch upscale images and calculate new GSD values
        """
        files = self.get_file_list()
        print(f"Found {len(files)} files to process in {self.input_folder}")
        
        upscale_results = []
        
        for i, file_path in enumerate(files, 1):
            print(f"\nProcessing {i}/{len(files)}: {file_path.name}")
            
            try:
                # Get original file info
                original_info, error = self.calculate_gsd_from_file(file_path)
                if error:
                    print(f"  Error: {error}")
                    continue
                
                # Define output path
                output_path = self.output_folder / f"upscaled_{file_path.stem}_{target_size[0]}x{target_size[1]}.tif"
                
                # Calculate new GSD after upscaling
                new_gsd_x = (original_info['gsd_x'] * original_info['width']) / target_size[0]
                new_gsd_y = (original_info['gsd_y'] * original_info['height']) / target_size[1]
                
                # Perform upscaling using GDAL Warp
                warp_options = gdal.WarpOptions(
                    format='GTiff',
                    width=target_size[0],
                    height=target_size[1],
                    resampleAlg=resample_method,
                    creationOptions=['COMPRESS=LZW', 'TILED=YES']
                )
                
                result = gdal.Warp(str(output_path), str(file_path), options=warp_options)
                
                if result:
                    # Verify the upscaled file
                    upscaled_info, error = self.calculate_gsd_from_file(output_path)
                    
                    if upscaled_info:
                        result_entry = {
                            'original_file': file_path.name,
                            'upscaled_file': output_path.name,
                            'original_size': f"{original_info['width']}x{original_info['height']}",
                            'upscaled_size': f"{upscaled_info['width']}x{upscaled_info['height']}",
                            'original_gsd': original_info['gsd_avg'],
                            'upscaled_gsd': upscaled_info['gsd_avg'],
                            'calculated_new_gsd': (new_gsd_x + new_gsd_y) / 2,
                            'resolution_improvement': original_info['gsd_avg'] / upscaled_info['gsd_avg'],
                            'geographic_area_preserved': abs(original_info['total_area_sqm'] - upscaled_info['total_area_sqm']) < 1.0,
                            'processing_status': 'Success'
                        }
                        
                        upscale_results.append(result_entry)
                        
                        print(f"  ✓ Original GSD: {original_info['gsd_avg']:.2f} m/px")
                        print(f"  ✓ New GSD: {upscaled_info['gsd_avg']:.2f} m/px")
                        print(f"  ✓ Resolution improvement: {result_entry['resolution_improvement']:.2f}x")
                        print(f"  ✓ Saved to: {output_path.name}")
                    
                    result = None
                else:
                    print(f"  ✗ Failed to upscale {file_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {file_path.name}: {str(e)}")
                upscale_results.append({
                    'original_file': file_path.name,
                    'processing_status': f'Error: {str(e)}'
                })
        
        return upscale_results
    
    def analyze_folder_gsd_statistics(self):
        """
        Analyze GSD statistics for all files in the folder
        """
        files = self.get_file_list()
        print(f"Analyzing GSD statistics for {len(files)} files...")
        
        for file_path in files:
            file_info, error = self.calculate_gsd_from_file(file_path)
            
            if file_info:
                self.results.append(file_info)
            else:
                print(f"Skipping {file_path.name}: {error}")
        
        return self.results
    
    def generate_summary_report(self, upscale_results=None):
        """
        Generate comprehensive summary report
        """
        # Analyze original files
        original_stats = self.analyze_folder_gsd_statistics()
        
        if not original_stats:
            print("No valid files found for analysis")
            return
        
        # Create DataFrame for analysis
        df = pd.DataFrame(original_stats)
        
        print("\n" + "="*80)
        print("BATCH GSD ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nFOLDER: {self.input_folder}")
        print(f"TOTAL FILES PROCESSED: {len(original_stats)}")
        print(f"ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nGSD STATISTICS:")
        print(f"  Mean GSD: {df['gsd_avg'].mean():.2f} m/px")
        print(f"  Median GSD: {df['gsd_avg'].median():.2f} m/px")
        print(f"  Min GSD: {df['gsd_avg'].min():.2f} m/px")
        print(f"  Max GSD: {df['gsd_avg'].max():.2f} m/px")
        print(f"  Std Dev: {df['gsd_avg'].std():.2f} m/px")
        
        print(f"\nIMAGE DIMENSIONS:")
        print(f"  Average Width: {df['width'].mean():.0f} pixels")
        print(f"  Average Height: {df['height'].mean():.0f} pixels")
        print(f"  Total Geographic Area: {df['total_area_sqm'].sum()/1000000:.2f} km²")
        
        # GSD distribution
        gsd_ranges = [
            (0, 1, "Sub-meter"),
            (1, 5, "High resolution"),
            (5, 15, "Medium resolution"),
            (15, 50, "Low resolution"),
            (50, float('inf'), "Very low resolution")
        ]
        
        print(f"\nGSD DISTRIBUTION:")
        for min_gsd, max_gsd, category in gsd_ranges:
            count = len(df[(df['gsd_avg'] >= min_gsd) & (df['gsd_avg'] < max_gsd)])
            if count > 0:
                print(f"  {category}: {count} files")
        
        # Upscaling results if provided
        if upscale_results:
            print(f"\nUPSCALING RESULTS:")
            successful = len([r for r in upscale_results if r.get('processing_status') == 'Success'])
            print(f"  Successfully processed: {successful}/{len(upscale_results)} files")
            
            if successful > 0:
                success_results = [r for r in upscale_results if r.get('processing_status') == 'Success']
                avg_improvement = np.mean([r['resolution_improvement'] for r in success_results])
                print(f"  Average resolution improvement: {avg_improvement:.2f}x")
        
        # Save detailed results to CSV
        csv_path = self.output_folder / f"gsd_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")
        
        # Save upscaling results if available
        if upscale_results:
            upscale_df = pd.DataFrame(upscale_results)
            upscale_csv = self.output_folder / f"upscaling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            upscale_df.to_csv(upscale_csv, index=False)
            print(f"Upscaling results saved to: {upscale_csv}")

# Usage example and batch processing functions
def process_sentinel2_batch(input_folder, output_folder=None, target_size=(800, 800)):
    """
    Main function to process a batch of Sentinel-2 images
    """
    processor = BatchGSDProcessor(input_folder, output_folder)
    
    print("Starting batch processing of Sentinel-2 images...")
    print(f"Input folder: {input_folder}")
    print(f"Target upscale size: {target_size[0]}x{target_size[1]}")
    
    # Step 1: Analyze original files
    print("\n--- STEP 1: ANALYZING ORIGINAL FILES ---")
    original_results = processor.analyze_folder_gsd_statistics()
    
    # Step 2: Batch upscale
    print("\n--- STEP 2: BATCH UPSCALING ---")
    upscale_results = processor.batch_upscale_with_gsd_calculation(target_size)
    
    # Step 3: Generate comprehensive report
    print("\n--- STEP 3: GENERATING REPORT ---")
    processor.generate_summary_report(upscale_results)
    
    return processor, upscale_results

if __name__ == "__main__":
    # Example usage
    input_folder = "/data_img/temp_lr/"  # Replace with your folder path
    output_folder = "/data_img/LR/"           # Replace with your output folder
    
    # Process all Sentinel-2 images in the folder
    processor, results = process_sentinel2_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        target_size=(800, 800)
    )
    
    print("\nBatch processing completed!")
