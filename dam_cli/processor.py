"""
Video and image processing utilities for the DAM CLI Wrapper
"""
import cv2
import glob
import numpy as np
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union
import tempfile

from dam_cli.logger import log_info, log_warning


def extract_frames_from_video(
    video_path: str, 
    output_dir: Optional[str] = None,
    num_frames: int = 8,
    logger=None
) -> List[str]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames. If None, creates a temp directory
        num_frames: Number of frames to extract. Default is 8 for DAM
        logger: Logger instance
        
    Returns:
        List of paths to the extracted frames
    """
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    if logger:
        log_info(logger, f"Extracting frames from '{video_path}' to '{output_dir}'")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        if logger:
            log_warning(logger, f"Could not open video: {video_path}")
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if logger:
        log_info(logger, f"Video has {total_frames} frames. Extracting {num_frames} frames.")
    
    # Calculate frame indices to extract
    if total_frames <= num_frames:
        # If fewer frames than requested, duplicate some frames
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    else:
        # Otherwise, sample uniformly
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frame_paths = []
    
    # Read and save frames
    for i, frame_idx in enumerate(indices):
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            if logger:
                log_warning(logger, f"Failed to read frame {frame_idx}")
            continue
        
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
    
    # Release video
    cap.release()
    
    return frame_paths


def load_frames_from_directory(
    directory: str, 
    num_frames: int = 8,
    extensions: List[str] = ["jpg", "jpeg", "png"],
    logger=None
) -> List[str]:
    """
    Load frames from a directory.
    
    Args:
        directory: Directory containing image frames
        num_frames: Number of frames to load. Default is 8 for DAM
        extensions: List of valid file extensions
        logger: Logger instance
        
    Returns:
        List of paths to the selected frames
    """
    # Find all image files in the directory
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, f"*.{ext}")))
    
    # Sort by name
    image_paths.sort()
    
    if logger:
        log_info(logger, f"Found {len(image_paths)} image files in '{directory}'")
    
    if not image_paths:
        if logger:
            log_warning(logger, f"No image files found in '{directory}'")
        raise ValueError(f"No image files found in '{directory}'")
    
    # Select frames
    if len(image_paths) <= num_frames:
        # If fewer frames than requested, duplicate some frames
        indices = np.linspace(0, len(image_paths)-1, num_frames, dtype=int)
    else:
        # Otherwise, sample uniformly
        indices = np.linspace(0, len(image_paths)-1, num_frames, dtype=int)
    
    selected_paths = [image_paths[i] for i in indices]
    
    return selected_paths


def load_and_process_frames(
    frame_paths: List[str],
    logger=None
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Load and process image frames, preparing for DAM input.
    
    Args:
        frame_paths: List of paths to image frames
        logger: Logger instance
        
    Returns:
        Tuple of (list of PIL Images, list of PIL mask images)
    """
    # Load images
    images = []
    masks = []
    
    for path in frame_paths:
        try:
            img = Image.open(path).convert('RGB')
            images.append(img)
            
            # Create a blank mask for now, will be filled by SAM
            mask = Image.new('L', img.size, 0)
            masks.append(mask)
        except Exception as e:
            if logger:
                log_warning(logger, f"Failed to load image '{path}': {str(e)}")
            continue
    
    return images, masks


def save_output_text(
    text: str,
    output_path: str,
    logger=None
) -> None:
    """
    Save output text to a file.
    
    Args:
        text: Text to save
        output_path: Path to save the text
        logger: Logger instance
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        if logger:
            log_info(logger, f"Output text saved to '{output_path}'")
    except Exception as e:
        if logger:
            log_warning(logger, f"Failed to save output text: {str(e)}")
