"""
SAM (Segment Anything Model) utilities for the DAM CLI Wrapper
"""
import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
from typing import List, Optional, Tuple, Union

from dam_cli.logger import log_info


def load_sam_model(device: Optional[str] = None) -> Tuple[SamModel, SamProcessor]:
    """
    Load SAM model and processor.
    
    Args:
        device: Device to load the model on (e.g., 'cuda:0', 'cpu')
        
    Returns:
        Tuple of SAM model and processor
    """
    # If device not specified, use CUDA if available
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model and processor
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    
    return sam_model, sam_processor


def apply_sam(
    image: Image.Image, 
    sam_model: SamModel, 
    sam_processor: SamProcessor, 
    input_points: Optional[List[List[List[int]]]] = None, 
    input_boxes: Optional[List[List[List[int]]]] = None, 
    input_labels: Optional[List[List[int]]] = None
) -> np.ndarray:
    """
    Apply SAM to an image to generate a segmentation mask.
    
    Args:
        image: PIL Image
        sam_model: SAM model
        sam_processor: SAM processor
        input_points: List of lists of points, e.g., [[[x1, y1], [x2, y2]]]
        input_boxes: List of boxes, e.g., [[[x1, y1, x2, y2]]]
        input_labels: List of labels for points, e.g., [[1, 1]]
        
    Returns:
        Segmentation mask as a NumPy array
    """
    # Prepare inputs
    inputs = sam_processor(
        image, 
        input_points=input_points, 
        input_boxes=input_boxes,
        input_labels=input_labels,
        return_tensors="pt"
    ).to(sam_model.device)
    
    # Generate masks
    with torch.no_grad():
        outputs = sam_model(**inputs)
    
    # Process outputs
    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0]
    scores = outputs.iou_scores[0, 0]
    
    # Select the mask with the highest score
    mask_selection_index = scores.argmax()
    mask_np = masks[mask_selection_index].numpy()
    
    return mask_np


def extract_points_from_mask(mask_pil: Image.Image) -> np.ndarray:
    """
    Extract points from a mask image.
    
    Args:
        mask_pil: PIL Image mask
        
    Returns:
        NumPy array of points
    """
    # Select the first channel of the mask
    mask = np.asarray(mask_pil)[..., 0]
    
    # coords is in (y_arr, x_arr) format
    coords = np.nonzero(mask)
    
    # coords is in [(x, y), ...] format
    coords = np.stack((coords[1], coords[0]), axis=1)
    
    return coords


def denormalize_coordinates(
    coords: Union[List[int], List[List[int]]], 
    image_size: Tuple[int, int], 
    is_box: bool = False
) -> Union[List[int], List[List[int]]]:
    """
    Convert normalized coordinates (0-1) to pixel coordinates.
    
    Args:
        coords: Coordinates as [x, y] or [x1, y1, x2, y2] or a list of these
        image_size: Image dimensions as (width, height)
        is_box: Whether the coordinates represent a bounding box
        
    Returns:
        Denormalized coordinates
    """
    width, height = image_size
    
    # Handle a single set of coordinates
    if isinstance(coords[0], (int, float)):
        if is_box:
            # For boxes: [x1, y1, x2, y2]
            x1, y1, x2, y2 = coords
            return [
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ]
        else:
            # For points: [x, y]
            x, y = coords
            return [int(x * width), int(y * height)]
    
    # Handle a list of coordinates
    result = []
    for coord in coords:
        if is_box:
            x1, y1, x2, y2 = coord
            result.append([
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            ])
        else:
            x, y = coord
            result.append([int(x * width), int(y * height)])
    
    return result
