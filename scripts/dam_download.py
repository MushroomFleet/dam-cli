#!/usr/bin/env python3
"""
Utility script for downloading DAM models
"""
import argparse
import sys
import os

# Add parent directory to path to allow importing dam_cli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dam_cli.download import download_model, check_model_exists
from dam_cli.logger import setup_logging, log_info, log_error, log_warning

def parse_args():
    """Parse command line arguments for the model download utility"""
    parser = argparse.ArgumentParser(
        description="Download a DAM model from Hugging Face Hub"
    )
    
    parser.add_argument(
        "model_id",
        type=str,
        help="Hugging Face model ID (e.g., 'nvidia/DAM-3B-Video')"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the model to"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if model already exists"
    )
    
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/download.log",
        help="Path to log file"
    )
    
    return parser.parse_args()

def run():
    """Run the model download utility"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the download path
    model_name = args.model_id.split('/')[-1]
    download_path = os.path.join(args.output_dir, model_name)
    
    # Check if model already exists
    if not args.force and check_model_exists(download_path):
        log_info(logger, f"Model already exists at '{download_path}'. Use --force to download anyway.")
        return 0
    
    # Download the model
    log_info(logger, f"Downloading model '{args.model_id}' to '{download_path}'...")
    success = download_model(
        model_id=args.model_id,
        local_dir=download_path,
        logger=logger
    )
    
    if success:
        log_info(logger, f"Model downloaded successfully to '{download_path}'")
        return 0
    else:
        log_error(logger, f"Failed to download model '{args.model_id}'")
        return 1

if __name__ == "__main__":
    sys.exit(run())
