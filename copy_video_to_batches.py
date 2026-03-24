#!/usr/bin/env python3
"""
Script to copy a video file to latency_calibration folders in all batch directories.

Usage:
    python copy_video_to_batches.py <source_video_path> <parent_data_directory>

Example:
    python copy_video_to_batches.py /path/to/video.mp4 /home/icrlab/tactile_work_Wy/data/simple-1.1/
"""

import os
import shutil
import argparse
import glob
from pathlib import Path


def copy_video_to_batches(source_video, parent_dir):
    """
    Copy a video to latency_calibration folders in all batch directories.
    
    Args:
        source_video: Path to the source video file
        parent_dir: Parent directory containing batch_* folders
    """
    
    # Validate source video exists
    if not os.path.isfile(source_video):
        print(f"❌ Error: Source video not found: {source_video}")
        return False
    
    # Validate parent directory exists
    if not os.path.isdir(parent_dir):
        print(f"❌ Error: Parent directory not found: {parent_dir}")
        return False
    
    # Get video filename
    video_filename = os.path.basename(source_video)
    print(f"📹 Source video: {source_video}")
    print(f"📁 Parent directory: {parent_dir}")
    print(f"📄 Video filename: {video_filename}\n")
    
    # Find all batch_* directories
    batch_pattern = os.path.join(parent_dir, 'batch_*')
    batch_dirs = sorted(glob.glob(batch_pattern))
    
    if not batch_dirs:
        print(f"⚠️  No batch_* directories found in: {parent_dir}")
        return False
    
    print(f"Found {len(batch_dirs)} batch folder(s):\n")
    
    success_count = 0
    failed_updates = []
    
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        latency_calib_dir = os.path.join(batch_dir, 'latency_calibration')
        
        # Check if latency_calibration directory exists
        if not os.path.isdir(latency_calib_dir):
            print(f"  ⚠️  {batch_name}: latency_calibration folder not found")
            failed_updates.append((batch_name, "latency_calibration folder not found"))
            continue
        
        # Copy video to latency_calibration directory
        destination_video = os.path.join(latency_calib_dir, video_filename)
        
        try:
            shutil.copy2(source_video, destination_video)
            print(f"  ✅ {batch_name}: Video copied successfully")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {batch_name}: Failed to copy - {str(e)}")
            failed_updates.append((batch_name, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  ✅ Successful copies: {success_count}/{len(batch_dirs)}")
    
    if failed_updates:
        print(f"  ❌ Failed copies: {len(failed_updates)}/{len(batch_dirs)}")
        for batch_name, error in failed_updates:
            print(f"     - {batch_name}: {error}")
    
    print(f"{'='*60}")
    
    return len(failed_updates) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Copy a video to latency_calibration folders in all batch directories.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python copy_video_to_batches.py /path/to/video.mp4 /home/icrlab/tactile_work_Wy/data/simple-1.1/
  python copy_video_to_batches.py ./reference.mp4 ./data/experiment_1/
        """
    )
    
    parser.add_argument(
        'source_video',
        help='Path to the source video file'
    )
    
    parser.add_argument(
        'parent_directory',
        help='Parent directory containing batch_* folders'
    )
    
    args = parser.parse_args()
    
    # Expand user path (~)
    source_video = os.path.expanduser(args.source_video)
    parent_dir = os.path.expanduser(args.parent_directory)
    
    # Run the copy operation
    success = copy_video_to_batches(source_video, parent_dir)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
