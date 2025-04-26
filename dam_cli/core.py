"""
Core functionality for the DAM CLI Wrapper
"""
import ast
import os
from typing import Dict, List, Optional, Tuple, Union
import torch
from PIL import Image
import tqdm

from dam_cli.cli import validate_args
from dam_cli.logger import setup_logging, log_info, log_error, log_warning
from dam_cli.download import ensure_model_downloaded
from dam_cli.dam_utils import load_dam_model, parse_points_list, parse_boxes_list, get_description, print_streaming
from dam_cli.sam_utils import load_sam_model, apply_sam, denormalize_coordinates
from dam_cli.processor import extract_frames_from_video, load_frames_from_directory, load_and_process_frames, save_output_text
from dam_cli.visualization import save_visualization


def process_video(args) -> str:
    """
    Process a video with DAM to generate a description.
    
    Args:
        args: Command line arguments
        
    Returns:
        Generated description
    """
    # Set up logging
    logger = setup_logging(args.log_file)
    
    # Validate arguments
    try:
        validate_args(args)
    except ValueError as e:
        log_error(logger, str(e))
        return ""
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_info(logger, f"Using device: {device}")
    
    # Ensure model is downloaded
    model_path = ensure_model_downloaded(args.model_path, "models", logger)
    if not model_path:
        log_error(logger, f"Failed to locate or download model: {args.model_path}")
        return ""
    
    # Load SAM model
    log_info(logger, "Loading SAM model...")
    sam_model, sam_processor = load_sam_model(device)
    
    # Load DAM model
    log_info(logger, "Loading DAM model...")
    dam_model = load_dam_model(
        model_path=model_path,
        conv_mode=args.conv_mode,
        prompt_mode=args.prompt_mode,
        device=device,
        logger=logger
    )
    if not dam_model:
        log_error(logger, "Failed to load DAM model")
        return ""
    
    # Get frame paths
    if args.video_file:
        # Extract frames from video file
        log_info(logger, f"Processing video file: {args.video_file}")
        frame_paths = extract_frames_from_video(
            args.video_file,
            output_dir=args.output_dir,
            num_frames=8,  # DAM expects 8 frames for videos
            logger=logger
        )
    elif args.video_dir:
        # Load frames from directory
        log_info(logger, f"Loading frames from directory: {args.video_dir}")
        frame_paths = load_frames_from_directory(
            args.video_dir,
            num_frames=8,  # DAM expects 8 frames for videos
            logger=logger
        )
    else:
        log_error(logger, "Either video_file or video_dir must be provided")
        return ""
    
    # Load frames
    log_info(logger, "Loading frames...")
    images, masks = load_and_process_frames(frame_paths, logger)
    
    if not images:
        log_error(logger, "No valid frames found")
        return ""
    
    # Parse points or boxes
    use_points = not args.use_box
    
    if use_points and args.points_list:
        points_list = parse_points_list(args.points_list)
        boxes_list = None
        
        if not points_list:
            log_error(logger, "Failed to parse points_list")
            return ""
    elif args.boxes_list:
        boxes_list = parse_boxes_list(args.boxes_list)
        points_list = None
        
        if not boxes_list:
            log_error(logger, "Failed to parse boxes_list")
            return ""
    else:
        log_error(logger, "Either points_list or boxes_list must be provided")
        return ""
    
    # Apply SAM to generate masks for all frames
    log_info(logger, "Applying SAM to generate masks...")
    for i, image in enumerate(tqdm.tqdm(images)):
        # Handle normalized coordinates if needed
        if args.normalized_coords:
            if points_list:
                # Denormalize points for this frame
                points = points_list[min(i, len(points_list) - 1)]
                points = denormalize_coordinates(points, image.size, is_box=False)
                input_points = [points]
                input_labels = [[1] * len(points)]
                
                mask_np = apply_sam(
                    image,
                    sam_model,
                    sam_processor,
                    input_points=input_points,
                    input_labels=input_labels
                )
            else:  # boxes_list
                # Denormalize boxes for this frame
                box = boxes_list[min(i, len(boxes_list) - 1)]
                box = denormalize_coordinates(box, image.size, is_box=True)
                input_boxes = [[box]]
                
                mask_np = apply_sam(
                    image,
                    sam_model,
                    sam_processor,
                    input_boxes=input_boxes
                )
        else:
            if points_list:
                # Use points as is
                points = points_list[min(i, len(points_list) - 1)]
                input_points = [points]
                input_labels = [[1] * len(points)]
                
                mask_np = apply_sam(
                    image,
                    sam_model,
                    sam_processor,
                    input_points=input_points,
                    input_labels=input_labels
                )
            else:  # boxes_list
                # Use box as is
                box = boxes_list[min(i, len(boxes_list) - 1)]
                input_boxes = [[box]]
                
                mask_np = apply_sam(
                    image,
                    sam_model,
                    sam_processor,
                    input_boxes=input_boxes
                )
        
        # Convert mask to PIL Image
        masks[i] = Image.fromarray((mask_np * 255).astype('uint8'))
        
        # Save visualization if requested
        if args.output_image_path or args.output_dir:
            if args.output_image_path:
                # For single output, add frame index to filename
                base, ext = os.path.splitext(args.output_image_path)
                output_path = f"{base}_{i:02d}{ext}"
            else:
                # For directory output
                output_path = os.path.join(args.output_dir, f"frame_{i:02d}.png")
            
            # Save visualization
            vis_points = points if points_list else None
            vis_box = box if boxes_list else None
            
            save_visualization(
                image,
                masks[i],
                output_path,
                points=vis_points,
                boxes=vis_box
            )
    
    # Generate description
    log_info(logger, "Generating description...")
    streaming = not args.no_stream
    
    if streaming:
        description = ""
        for token in get_description(
            model=dam_model,
            images=images,
            masks=masks,
            query=args.query,
            streaming=True,
            temperature=args.temperature,
            top_p=args.top_p,
            logger=logger
        ):
            description += token
            print_streaming(token)
        
        print()  # Add newline at the end
    else:
        description = get_description(
            model=dam_model,
            images=images,
            masks=masks,
            query=args.query,
            streaming=False,
            temperature=args.temperature,
            top_p=args.top_p,
            logger=logger
        )
        print(description)
    
    # Save description if output path provided
    if args.output_text_path:
        save_output_text(description, args.output_text_path, logger)
    elif args.output_dir:
        save_output_text(description, os.path.join(args.output_dir, "description.txt"), logger)
    
    log_info(logger, "Processing complete")
    
    return description


def main(args) -> int:
    """
    Main entry point for the DAM CLI.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        description = process_video(args)
        return 0 if description else 1
    except Exception as e:
        logger = setup_logging(args.log_file if hasattr(args, 'log_file') else None)
        log_error(logger, f"Unexpected error: {str(e)}")
        return 1
