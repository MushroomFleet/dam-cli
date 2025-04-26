"""
DAM (Describe Anything Model) utilities for the DAM CLI Wrapper
"""
import ast
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union

# Using relative import to handle the case where the original dam package isn't installed
try:
    # Try to import from installed package
    from dam import DescribeAnythingModel, disable_torch_init, DEFAULT_IMAGE_TOKEN
except ImportError:
    print("Warning: Unable to import from installed 'dam' package.")
    # Set placeholders for required components
    DescribeAnythingModel = None
    DEFAULT_IMAGE_TOKEN = "<image>"
    
    def disable_torch_init():
        """Dummy function if original not available"""
        pass

from dam_cli.logger import log_info, log_error, log_warning


def load_dam_model(
    model_path: str,
    conv_mode: str = "v1",
    prompt_mode: str = "focal_prompt",
    device: Optional[str] = None,
    logger=None
) -> Optional[DescribeAnythingModel]:
    """
    Load DAM model.
    
    Args:
        model_path: Path to the model checkpoint
        conv_mode: Conversation mode
        prompt_mode: Prompt mode
        device: Device to load the model on (e.g., 'cuda:0', 'cpu')
        logger: Logger instance
        
    Returns:
        DAM model or None if loading fails
    """
    if DescribeAnythingModel is None:
        if logger:
            log_error(logger, "DAM package not available. Please install it first.")
        return None
    
    # If device not specified, use CUDA if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    if logger:
        log_info(logger, f"Loading DAM model from '{model_path}' on {device}...")
    
    try:
        # Disable torch init to speed up model loading
        disable_torch_init()
        
        # Map prompt mode to internal format
        prompt_modes = {
            "focal_prompt": "full+focal_crop",
        }
        internal_prompt_mode = prompt_modes.get(prompt_mode, prompt_mode)
        
        # Load model
        dam_model = DescribeAnythingModel(
            model_path=model_path,
            conv_mode=conv_mode,
            prompt_mode=internal_prompt_mode,
        ).to(device)
        
        if logger:
            log_info(logger, f"Model loaded successfully.")
        
        return dam_model
    except Exception as e:
        if logger:
            log_error(logger, f"Failed to load DAM model: {str(e)}")
        return None


def parse_points_list(points_list_str: Optional[str]) -> Optional[List[List[List[int]]]]:
    """
    Parse points list string into proper format.
    
    Args:
        points_list_str: String representation of points list
        
    Returns:
        Parsed points list or None if parsing fails
    """
    if not points_list_str:
        return None
    
    try:
        # Parse the string to Python object
        points_list = ast.literal_eval(points_list_str)
        
        # If a single set of points is provided, wrap it as a list of one element
        if isinstance(points_list[0][0], (int, float)):
            points_list = [points_list]
        
        return points_list
    except (SyntaxError, ValueError) as e:
        return None


def parse_boxes_list(boxes_list_str: Optional[str]) -> Optional[List[List[int]]]:
    """
    Parse boxes list string into proper format.
    
    Args:
        boxes_list_str: String representation of boxes list
        
    Returns:
        Parsed boxes list or None if parsing fails
    """
    if not boxes_list_str:
        return None
    
    try:
        # Parse the string to Python object
        boxes_list = ast.literal_eval(boxes_list_str)
        
        # If a single box is provided, wrap it as a list of one element
        if isinstance(boxes_list[0], (int, float)):
            boxes_list = [boxes_list]
        
        return boxes_list
    except (SyntaxError, ValueError) as e:
        return None


def get_description(
    model: DescribeAnythingModel,
    images: List[Image.Image],
    masks: List[Image.Image],
    query: str,
    streaming: bool = True,
    temperature: float = 0.2,
    top_p: float = 0.5,
    num_beams: int = 1,
    max_new_tokens: int = 512,
    logger=None
) -> Union[str, List[str]]:
    """
    Get description from DAM model.
    
    Args:
        model: DAM model
        images: List of images
        masks: List of masks
        query: Query string
        streaming: Whether to stream tokens
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_beams: Number of beams for beam search
        max_new_tokens: Maximum number of tokens to generate
        logger: Logger instance
        
    Returns:
        Description string or list of tokens if streaming
    """
    if logger:
        log_info(logger, f"Generating description for {len(images)} images...")
    
    # Remove prefix of query if it exists
    query = query.strip()
    query = query.removeprefix("Image:")
    query = query.removeprefix("Video:")
    query = query.strip()
    
    while query.startswith(DEFAULT_IMAGE_TOKEN):
        query = query.removeprefix(DEFAULT_IMAGE_TOKEN)
    
    query = query.strip()
    
    # Determine if we're processing video or image based on number of frames
    if len(images) == 8:  # Video mode
        query = f"Video: {DEFAULT_IMAGE_TOKEN * 8}\n{query}"
    elif len(images) == 1:  # Image mode
        query = f"{DEFAULT_IMAGE_TOKEN}\n{query}"
    else:
        if logger:
            log_warning(logger, f"Unusual number of frames ({len(images)}). Expected 1 for image or 8 for video.")
        # Try to handle it gracefully
        query = f"{DEFAULT_IMAGE_TOKEN * len(images)}\n{query}"
    
    try:
        # Generate description
        result = model.get_description(
            images,
            masks,
            query,
            streaming=streaming,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        
        if streaming:
            # If streaming, return generator as is
            return result
        else:
            # If not streaming, return the full string
            return result
    except Exception as e:
        if logger:
            log_error(logger, f"Error generating description: {str(e)}")
        return "" if not streaming else []


def print_streaming(text: str, end: str = "", flush: bool = True) -> None:
    """Helper function to print streaming text with flush"""
    print(text, end=end, flush=flush)
