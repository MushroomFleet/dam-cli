"""
Command line argument parsing for DAM CLI Wrapper
"""
import argparse
import os
from typing import Any, Dict, Optional


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the DAM CLI tool"""
    parser = argparse.ArgumentParser(
        description="Describe Anything Model (DAM) CLI for video captioning"
    )
    
    # Main input parameters
    parser.add_argument(
        "--video_dir", 
        type=str, 
        help="Directory containing video frames"
    )
    parser.add_argument(
        "--video_file",
        type=str,
        help="Video file path (will be extracted to frames)"
    )
    parser.add_argument(
        "--points_list",
        type=str,
        help="List of points for each frame, format: [[[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]], ...]"
    )
    parser.add_argument(
        "--boxes_list",
        type=str,
        help="List of boxes for each frame, format: [[x1,y1,x2,y2], [x3,y3,x4,y4], ...]"
    )
    
    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="nvidia/DAM-3B-Video",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="focal_prompt",
        help="Prompt mode"
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="v1",
        help="Conversation mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.5,
        help="Top-p for sampling"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_image_path",
        type=str,
        help="Path to save the output image with contour"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save output images and text"
    )
    parser.add_argument(
        "--output_text_path",
        type=str,
        help="Path to save the output text description"
    )
    
    # Behavior flags
    parser.add_argument(
        "--normalized_coords",
        action="store_true",
        help="Interpret coordinates as normalized (0-1) values"
    )
    parser.add_argument(
        "--no_stream",
        action="store_true",
        help="Disable streaming output"
    )
    parser.add_argument(
        "--use_box",
        action="store_true",
        help="Use bounding boxes instead of points"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/processing.log",
        help="Path to log file"
    )
    
    # Query/prompt customization
    parser.add_argument(
        "--query",
        type=str,
        default="Video: <image><image><image><image><image><image><image><image>\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.",
        help="Prompt for the model"
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments for the DAM CLI tool"""
    # Check that either video_dir or video_file is provided
    if not args.video_dir and not args.video_file:
        raise ValueError("Either --video_dir or --video_file must be specified")
    
    # Check that at least one of points_list or boxes_list is provided
    if not args.points_list and not args.boxes_list and not args.use_box:
        raise ValueError("Either --points_list or --boxes_list must be specified")
    
    # Create output directory if it doesn't exist
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create log directory if it doesn't exist
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)


def args_to_dict(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse Namespace to dictionary"""
    return vars(args)


def dict_to_args(arg_dict: Dict[str, Any]) -> argparse.Namespace:
    """Convert dictionary to argparse Namespace"""
    args = argparse.Namespace()
    for key, value in arg_dict.items():
        setattr(args, key, value)
    return args
