"""
Model download utilities for the DAM CLI Wrapper
"""
import os
from typing import Optional

import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from dam_cli.logger import log_info, log_error, log_warning


def download_model(model_id: str, local_dir: str, logger=None) -> bool:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        model_id: Hugging Face model ID (e.g., 'nvidia/DAM-3B-Video')
        local_dir: Local directory to download the model to
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if logger:
            log_info(logger, f"Downloading model '{model_id}' to '{local_dir}'...")
        
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download model
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            tqdm_class=tqdm.tqdm
        )
        
        if logger:
            log_info(logger, f"Model downloaded successfully to '{local_dir}'")
        
        return True
    except HfHubHTTPError as e:
        if logger:
            log_error(logger, f"Error downloading model: {str(e)}")
        return False
    except Exception as e:
        if logger:
            log_error(logger, f"Unexpected error downloading model: {str(e)}")
        return False


def check_model_exists(model_path: str) -> bool:
    """
    Check if a model exists at the specified path.
    
    Args:
        model_path: Path to the model
        
    Returns:
        True if the model exists, False otherwise
    """
    # Check if the path exists and contains model files
    required_files = ['config.json']
    
    if not os.path.exists(model_path):
        return False
    
    # Check if required files exist
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    
    return True


def resolve_model_path(model_id_or_path: str, models_dir: str = "models") -> str:
    """
    Resolve a model ID or path to a local path.
    
    If the input is a valid path to an existing model, return it.
    If the input is a model ID, return the path where it would be downloaded.
    
    Args:
        model_id_or_path: Hugging Face model ID (e.g., 'nvidia/DAM-3B-Video') or path
        models_dir: Base directory for models
        
    Returns:
        Resolved local path
    """
    # If it's a full path and exists, return it
    if os.path.exists(model_id_or_path) and check_model_exists(model_id_or_path):
        return model_id_or_path
    
    # If it's a relative path under models_dir
    local_path = os.path.join(models_dir, model_id_or_path)
    if os.path.exists(local_path) and check_model_exists(local_path):
        return local_path
    
    # If it's a Hugging Face model ID
    model_name = model_id_or_path.split('/')[-1]
    local_path = os.path.join(models_dir, model_name)
    
    return local_path


def ensure_model_downloaded(model_id_or_path: str, models_dir: str = "models", logger=None) -> Optional[str]:
    """
    Ensure a model is downloaded and available.
    
    Args:
        model_id_or_path: Hugging Face model ID or path
        models_dir: Base directory for models
        logger: Logger instance
        
    Returns:
        Path to the model if successful, None otherwise
    """
    # First, try to resolve as a local path
    local_path = resolve_model_path(model_id_or_path, models_dir)
    
    # If the model exists locally, return the path
    if check_model_exists(local_path):
        if logger:
            log_info(logger, f"Using existing model at '{local_path}'")
        return local_path
    
    # Otherwise, try to download
    if '/' in model_id_or_path:  # Assume it's a Hugging Face model ID
        if logger:
            log_info(logger, f"Model not found locally. Downloading '{model_id_or_path}'...")
        
        success = download_model(model_id_or_path, local_path, logger)
        
        if success:
            return local_path
        else:
            if logger:
                log_error(logger, f"Failed to download model '{model_id_or_path}'")
            return None
    else:
        if logger:
            log_warning(logger, f"Model not found at '{local_path}' and doesn't look like a Hugging Face model ID")
        return None
