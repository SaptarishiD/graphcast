import os
from pathlib import Path
import xarray
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class StatisticsFiles:
    diffs_stddev_by_level: xarray.Dataset
    mean_by_level: xarray.Dataset
    stddev_by_level: xarray.Dataset

def download_statistics(
    gcs_bucket,
    dir_prefix: str,
    local_dir: str = "stats",
    force_download: bool = False
) -> None:
    """
    Download statistics files from GCS bucket to local directory.
    
    Args:
        gcs_bucket: Google Cloud Storage bucket
        dir_prefix: Prefix for files in the bucket
        local_dir: Local directory to save files to
        force_download: Whether to download files even if they exist locally
    """
    # Create local directory if it doesn't exist
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # Files to download
    files = [
        "diffs_stddev_by_level.nc",
        "mean_by_level.nc",
        "stddev_by_level.nc"
    ]
    
    # Download each file
    for filename in files:
        local_file = local_path / filename
        
        # Skip if file exists and force_download is False
        if local_file.exists() and not force_download:
            print(f"Skipping {filename} (already exists)")
            continue
            
        print(f"Downloading {filename}...")
        with local_file.open("wb") as f_out:
            with gcs_bucket.blob(dir_prefix + f"stats/{filename}").open("rb") as f_in:
                f_out.write(f_in.read())

def load_statistics(local_dir: str = "stats") -> StatisticsFiles:
    """
    Load statistics from local files.
    
    Args:
        local_dir: Directory containing the statistics files
    
    Returns:
        StatisticsFiles object containing loaded datasets
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Directory {local_dir} not found")
        
    try:
        diffs_stddev = xarray.load_dataset(local_path / "diffs_stddev_by_level.nc").compute()
        means = xarray.load_dataset(local_path / "mean_by_level.nc").compute()
        stddev = xarray.load_dataset(local_path / "stddev_by_level.nc").compute()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing statistics files in {local_dir}. "
            "Run download_statistics() first."
        ) from e
        
    return StatisticsFiles(diffs_stddev, means, stddev)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and load GraphCast statistics")
    parser.add_argument("--bucket", required=True, help="GCS bucket name")
    parser.add_argument("--prefix", required=True, help="Directory prefix in bucket")
    parser.add_argument("--local-dir", default="stats", help="Local directory for files")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    
    args = parser.parse_args()
    
    # Initialize GCS bucket (assuming google-cloud-storage is imported)
    from google.cloud import storage
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(args.bucket)
    # gcs_bucket = gcs_client.get_bucket("dm_graphcast")
    # dir_prefix = "graphcast/"
    # Download statistics
    download_statistics(bucket, args.prefix, args.local_dir, args.force)
    
    # Load and print basic info about the statistics
    stats = load_statistics(args.local_dir)
    print("\nStatistics loaded successfully:")
    print(f"Diffs StdDev shape: {stats.diffs_stddev_by_level.dims}")
    print(f"Means shape: {stats.mean_by_level.dims}")
    print(f"StdDev shape: {stats.stddev_by_level.dims}")