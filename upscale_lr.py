#!/usr/bin/env python3
"""
Batch processor for calculating GSD and upscaling GeoTIFFs (uses rasterio, not osgeo).
"""

import os
import math
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling
from rasterio.transform import Affine


def _resampling_from_str(name: str) -> Resampling:
    name = (name or "").lower()
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
    }.get(name, Resampling.cubic)  # default cubic to match your prior code


class BatchGSDProcessor:
    """
    Batch processor for calculating and verifying GSD across multiple GeoTIFF files (rasterio).
    """

    def __init__(self, input_folder, output_folder=None):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder) if output_folder else self.input_folder / "processed"
        self.results = []
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def get_file_list(self, pattern="*.tif"):
        """
        Get list of GeoTIFF files in the input folder
        """
        file_patterns = [pattern, "*.tiff", "*.TIF", "*.TIFF"]
        files = []
        for pat in file_patterns:
            files.extend(list(self.input_folder.glob(pat)))
        return sorted(files)

    def calculate_gsd_from_file(self, file_path):
        """
        Calculate GSD and basic metadata from a single GeoTIFF
        """
        try:
            with rasterio.open(file_path) as ds:
                transform = ds.transform  # Affine(a, b, c, d, e, f)
                width, height, bands = ds.width, ds.height, ds.count
                # Pixel size in projected units (usually meters if CRS is projected)
                gsd_x = abs(transform.a)
                gsd_y = abs(transform.e)

                crs = ds.crs
                epsg = crs.to_epsg() if crs else None

                # Extent (minx, maxy are at origin; maxx/miny from size and pixel size & rotation terms)
                # For north-up rasters (b=d=0), this is straightforward:
                min_x = transform.c
                max_y = transform.f
                max_x = min_x + transform.a * width
                min_y = max_y + transform.e * height

                file_info = {
                    "filename": Path(file_path).name,
                    "filepath": str(file_path),
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "gsd_x": gsd_x,
                    "gsd_y": gsd_y,
                    "gsd_avg": (gsd_x + gsd_y) / 2.0,
                    "geographic_width": width * gsd_x,
                    "geographic_height": height * gsd_y,
                    "total_area_sqm": (width * gsd_x) * (height * gsd_y),
                    "extent": {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y},
                    "projection": f"EPSG:{epsg}" if epsg is not None else (str(crs) if crs else "Unknown"),
                    "file_size_mb": Path(file_path).stat().st_size / (1024 * 1024),
                    "dtype": ds.dtypes[0],
                    "nodata": ds.nodata,
                }
                return file_info, None
        except Exception as e:
            return None, f"Error processing {file_path}: {e}"

    def _upscale_one(self, src_path, dst_path, target_size, resample_method="cubic"):
        """
        Upscale to exact (width, height) using rasterio.warp.reproject per-band.
        Preserves geographic extent; pixel size changes accordingly (new GSD).
        """
        target_w, target_h = target_size
        resamp = _resampling_from_str(resample_method)

        with rasterio.open(src_path) as src:
            src_transform = src.transform
            src_crs = src.crs
            count = src.count
            dtype = src.dtypes[0]
            nodata = src.nodata

            # Compute new transform that spans the same extent but with new pixel sizes
            sx = src.width / float(target_w)
            sy = src.height / float(target_h)
            dst_transform = Affine(src_transform.a * sx, src_transform.b, src_transform.c,
                                   src_transform.d, src_transform.e * sy, src_transform.f)

            profile = src.profile.copy()
            profile.update({
                "width": target_w,
                "height": target_h,
                "transform": dst_transform,
                "dtype": dtype,
                "count": count,
                "nodata": nodata,
                "compress": "LZW",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "driver": "GTiff",
            })

            with rasterio.open(dst_path, "w", **profile) as dst:
                for b in range(1, count + 1):
                    src_band = src.read(b)
                    dst_band = np.empty((target_h, target_w), dtype=dtype)
                    reproject(
                        source=src_band,
                        destination=dst_band,
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=src_crs,
                        resampling=resamp,
                        src_nodata=nodata,
                        dst_nodata=nodata,
                    )
                    dst.write(dst_band, b)

    def batch_upscale_with_gsd_calculation(self, target_size=(800, 800), resample_method="cubic"):
        """
        Batch upscale images and calculate new GSD values (rasterio)
        """
        files = self.get_file_list()
        print(f"Found {len(files)} files to process in {self.input_folder}")
        results = []

        for i, file_path in enumerate(files, 1):
            print(f"\nProcessing {i}/{len(files)}: {Path(file_path).name}")

            try:
                original_info, error = self.calculate_gsd_from_file(file_path)
                if error:
                    print(f"  Error: {error}")
                    continue

                out_name = f"upscaled_{Path(file_path).stem}_{target_size[0]}x{target_size[1]}.tif"
                out_path = self.output_folder / out_name

                # New GSD expected (preserve extent; change pixel size)
                new_gsd_x = (original_info["gsd_x"] * original_info["width"]) / target_size[0]
                new_gsd_y = (original_info["gsd_y"] * original_info["height"]) / target_size[1]

                # Do the upscale
                self._upscale_one(file_path, out_path, target_size, resample_method)

                # Verify
                upscaled_info, error2 = self.calculate_gsd_from_file(out_path)
                if upscaled_info:
                    result = {
                        "original_file": Path(file_path).name,
                        "upscaled_file": out_name,
                        "original_size": f"{original_info['width']}x{original_info['height']}",
                        "upscaled_size": f"{upscaled_info['width']}x{upscaled_info['height']}",
                        "original_gsd": original_info["gsd_avg"],
                        "upscaled_gsd": upscaled_info["gsd_avg"],
                        "calculated_new_gsd": (new_gsd_x + new_gsd_y) / 2.0,
                        "resolution_improvement": original_info["gsd_avg"] / upscaled_info["gsd_avg"]
                        if upscaled_info["gsd_avg"] else np.nan,
                        "geographic_area_preserved": abs(
                            original_info["total_area_sqm"] - upscaled_info["total_area_sqm"]
                        ) < 1.0,
                        "processing_status": "Success",
                    }
                    results.append(result)
                    print(f"  ✓ Original GSD: {original_info['gsd_avg']:.6f}")
                    print(f"  ✓ New GSD: {upscaled_info['gsd_avg']:.6f}")
                    print(f"  ✓ Resolution improvement: {result['resolution_improvement']:.3f}x")
                    print(f"  ✓ Saved to: {out_name}")
                else:
                    print(f"  ✗ Failed to verify {out_name}: {error2}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({"original_file": Path(file_path).name, "processing_status": f"Error: {e}"})

        return results

    def analyze_folder_gsd_statistics(self):
        """
        Analyze GSD statistics for all files in the folder
        """
        files = self.get_file_list()
        print(f"Analyzing GSD statistics for {len(files)} files...")

        for file_path in files:
            info, err = self.calculate_gsd_from_file(file_path)
            if info:
                self.results.append(info)
            else:
                print(f"Skipping {Path(file_path).name}: {err}")
        return self.results

    def generate_summary_report(self, upscale_results=None):
        """
        Generate comprehensive summary report & save CSVs
        """
        original_stats = self.analyze_folder_gsd_statistics()
        if not original_stats:
            print("No valid files found for analysis")
            return

        df = pd.DataFrame(original_stats)

        print("\n" + "=" * 80)
        print("BATCH GSD ANALYSIS REPORT")
        print("=" * 80)
        print(f"\nFOLDER: {self.input_folder}")
        print(f"TOTAL FILES PROCESSED: {len(original_stats)}")
        print(f"ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\nGSD STATISTICS:")
        print(f"  Mean GSD: {df['gsd_avg'].mean():.6f}")
        print(f"  Median GSD: {df['gsd_avg'].median():.6f}")
        print(f"  Min GSD: {df['gsd_avg'].min():.6f}")
        print(f"  Max GSD: {df['gsd_avg'].max():.6f}")
        print(f"  Std Dev: {df['gsd_avg'].std():.6f}")

        print("\nIMAGE DIMENSIONS:")
        print(f"  Average Width: {df['width'].mean():.0f} pixels")
        print(f"  Average Height: {df['height'].mean():.0f} pixels")
        print(f"  Total Geographic Area: {df['total_area_sqm'].sum() / 1_000_000:.2f} km²")

        # GSD distribution
        gsd_ranges = [
            (0, 1, "Sub-meter"),
            (1, 5, "High resolution"),
            (5, 15, "Medium resolution"),
            (15, 50, "Low resolution"),
            (50, float("inf"), "Very low resolution"),
        ]
        print("\nGSD DISTRIBUTION:")
        for lo, hi, label in gsd_ranges:
            count = len(df[(df["gsd_avg"] >= lo) & (df["gsd_avg"] < hi)])
            if count > 0:
                print(f"  {label}: {count} files")

        if upscale_results is not None:
            print("\nUPSCALING RESULTS:")
            successful = len([r for r in upscale_results if r.get("processing_status") == "Success"])
            total = len(upscale_results)
            print(f"  Successfully processed: {successful}/{total} files")
            if successful > 0:
                success_rows = [r for r in upscale_results if r.get("processing_status") == "Success"]
                avg_improvement = float(np.mean([r["resolution_improvement"] for r in success_rows]))
                print(f"  Average resolution improvement: {avg_improvement:.3f}x")

        # Save detailed results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_folder / f"gsd_analysis_{ts}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to: {csv_path}")

        if upscale_results is not None:
            upscale_df = pd.DataFrame(upscale_results)
            upscale_csv = self.output_folder / f"upscaling_results_{ts}.csv"
            upscale_df.to_csv(upscale_csv, index=False)
            print(f"Upscaling results saved to: {upscale_csv}")


# -------- Batch entrypoint (kept similar to your example) --------
def process_sentinel2_batch(input_folder, output_folder=None, target_size=(800, 800), resample_method="cubic"):
    """
    Main function to process a batch of images
    """
    processor = BatchGSDProcessor(input_folder, output_folder)

    print("Starting batch processing of Sentinel-2 images...")
    print(f"Input folder: {input_folder}")
    print(f"Target upscale size: {target_size[0]}x{target_size[1]}")
    print(f"Resampling: {resample_method}")

    print("\n--- STEP 1: ANALYZING ORIGINAL FILES ---")
    _ = processor.analyze_folder_gsd_statistics()

    print("\n--- STEP 2: BATCH UPSCALING ---")
    upscale_results = processor.batch_upscale_with_gsd_calculation(target_size, resample_method)

    print("\n--- STEP 3: GENERATING REPORT ---")
    processor.generate_summary_report(upscale_results)

    return processor, upscale_results


if __name__ == "__main__":
    # Example usage (adjust paths to your environment)
    input_folder = "/data_img/temp_lr/"
    output_folder = "/data_img/LR/"
    processor, results = process_sentinel2_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        target_size=(800, 800),
        resample_method="cubic",
    )
    print("\nBatch processing completed!")