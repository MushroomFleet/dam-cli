"""
Visualization utilities for the DAM CLI Wrapper
"""
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union


def add_contour(
    img: np.ndarray, 
    mask: np.ndarray, 
    input_points: Optional[List[List[List[int]]]] = None, 
    input_boxes: Optional[List[List[List[int]]]] = None,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    point_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    box_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    thickness: int = 6,
    point_radius: int = 10
) -> np.ndarray:
    """
    Add contour and input points or boxes to an image.
    
    Args:
        img: Image as NumPy array (float 0-1)
        mask: Binary mask as NumPy array (0 or 1)
        input_points: List of lists of points
        input_boxes: List of boxes
        color: Contour color as RGB tuple (0-1)
        point_color: Point marker color as RGB tuple (0-1)
        box_color: Box color as RGB tuple (0-1)
        thickness: Line thickness
        point_radius: Radius of point markers
        
    Returns:
        Image with contour and markers as NumPy array
    """
    # Make a copy to avoid modifying the input
    img = img.copy()
    
    # Convert mask to 8-bit format
    mask = mask.astype(np.uint8) * 255
    
    # Find and draw contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness=thickness)
    
    # Draw points if provided
    if input_points is not None:
        for points in input_points:  # Handle batch of points
            for x, y in points:
                # Draw a filled circle for each point
                cv2.circle(img, (int(x), int(y)), 
                           radius=point_radius, 
                           color=point_color, 
                           thickness=-1)
                # Draw a white border around the circle
                cv2.circle(img, (int(x), int(y)), 
                           radius=point_radius, 
                           color=(1.0, 1.0, 1.0), 
                           thickness=2)
    
    # Draw boxes if provided
    if input_boxes is not None:
        for box_batch in input_boxes:  # Handle batch of boxes
            for box in box_batch:  # Iterate through boxes in the batch
                x1, y1, x2, y2 = map(int, box)
                # Draw rectangle with white color (border)
                cv2.rectangle(img, (x1, y1), (x2, y2), 
                              color=(1.0, 1.0, 1.0), 
                              thickness=thickness)
                # Draw inner rectangle with specified color
                cv2.rectangle(img, (x1, y1), (x2, y2), 
                              color=box_color, 
                              thickness=thickness-2)
    
    return img


def create_visualization(
    image: Union[np.ndarray, Image.Image],
    mask: Union[np.ndarray, Image.Image],
    points: Optional[List[List[int]]] = None,
    boxes: Optional[List[List[int]]] = None
) -> Image.Image:
    """
    Create visualization with mask contour and input points or boxes.
    
    Args:
        image: Input image (PIL Image or NumPy array)
        mask: Mask (PIL Image or NumPy array)
        points: List of points
        boxes: List of boxes
        
    Returns:
        Visualization as PIL Image
    """
    # Convert inputs to NumPy arrays if needed
    if isinstance(image, Image.Image):
        img_np = np.asarray(image).astype(float) / 255.0
    else:
        img_np = image.astype(float) / 255.0
    
    if isinstance(mask, Image.Image):
        mask_np = np.asarray(mask)[..., 0] if mask.mode == 'RGB' else np.asarray(mask)
        mask_np = (mask_np > 0).astype(np.uint8)
    else:
        mask_np = (mask > 0).astype(np.uint8)
    
    # Format points and boxes for add_contour
    input_points = [[points]] if points is not None else None
    input_boxes = [[boxes]] if boxes is not None else None
    
    # Create visualization
    vis_np = add_contour(img_np, mask_np, input_points, input_boxes)
    
    # Convert to PIL Image
    vis_pil = Image.fromarray((vis_np * 255.0).astype(np.uint8))
    
    return vis_pil


def save_visualization(
    image: Union[np.ndarray, Image.Image],
    mask: Union[np.ndarray, Image.Image],
    output_path: str,
    points: Optional[List[List[int]]] = None,
    boxes: Optional[List[List[int]]] = None
) -> None:
    """
    Create and save visualization with mask contour and input points or boxes.
    
    Args:
        image: Input image (PIL Image or NumPy array)
        mask: Mask (PIL Image or NumPy array)
        output_path: Path to save the visualization
        points: List of points
        boxes: List of boxes
    """
    vis_pil = create_visualization(image, mask, points, boxes)
    vis_pil.save(output_path)
