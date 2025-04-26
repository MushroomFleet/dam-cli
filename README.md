# DAM CLI: Describe Anything Model Command Line Interface

A standalone wrapper for NVIDIA's Describe Anything Model (DAM), focusing on the video captioning functionality.

## Overview

The DAM CLI is a user-friendly command-line tool that provides access to NVIDIA's powerful Describe Anything Model (DAM), with a focus on video captioning. This tool allows you to easily generate detailed descriptions of specific regions in videos by leveraging the DAM model in combination with the Segment Anything Model (SAM) for precise object segmentation.

The CLI wraps around the functionality provided in the original `dam_video_with_sam.py` example script from the Describe Anything Model repository, making it more accessible, configurable, and robust.

## Features

- **Video Processing**: Generate detailed descriptions for specific regions in videos
- **Image/Frame Selection**: Process video files or directories containing video frames
- **Object Selection**: Target specific objects using points or bounding boxes
- **Model Management**: Automatic downloading and management of models
- **User-friendly CLI**: Simple and intuitive command-line interface
- **Configuration Files**: Support for configuration files for repetitive tasks
- **Visualization**: Generate visualizations of the segmented regions
- **Streaming Output**: View description tokens as they're generated

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Windows

```bash
# Clone the repository
git clone https://github.com/MushroomFleet/dam-cli
cd dam-cli

# Run the installation script (dam-cli)
install.bat

# NOTE: Required DAM package
cd ..
git clone https://github.com/MushroomFleet/describe-anything
cd describe-anything

# Run the installation script (DAM)
install.bat
```

### Linux/Mac

```bash
# Clone the dam-cli wrapper
git clone https://github.com/MushroomFleet/dam-cli
cd dam-cli

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh

# For GPU acceleration (CUDA)
pip install -r requirements-cuda.txt

# DAM package

git clone https://github.com/MushroomFleet/describe-anything
cd describe-anything

# For GPU acceleration (CUDA)
pip install -v .
```

### Manual Installation

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

## Model Setup

Before using DAM CLI, you need to download the required models:

```bash
# Download DAM-3B-Video model from Hugging Face
python scripts/dam_download.py nvidia/DAM-3B-Video
```

The model will be downloaded to the `models` directory.

## Usage Examples

### Basic Usage with Points

```bash
python scripts/dam_describe.py --video_file path/to/video.mp4 --points_list "[[[1172, 812], [1572, 800]]]"
```

### Using Bounding Boxes

```bash
python scripts/dam_describe.py --video_file path/to/video.mp4 --boxes_list "[[800, 500, 1800, 1000]]" --use_box
```

### Processing a Directory of Frames

```bash
python scripts/dam_describe.py --video_dir path/to/frames --points_list "[[[1172, 812], [1572, 800]]]"
```

### Using a Configuration File

```bash
python scripts/dam_describe.py --config path/to/config.json
```

### Saving Output

```bash
python scripts/dam_describe.py --video_file path/to/video.mp4 --points_list "[[[1172, 812], [1572, 800]]]" --output_dir path/to/output
```

## Configuration File Format

You can use a JSON configuration file to avoid typing long commands. Here's an example:

```json
{
    "video_dir": "path/to/video/frames",
    "points_list": [[[1172, 812], [1572, 800]]],
    "query": "Video: <image><image><image><image><image><image><image><image>\nGiven the video in the form of a sequence of frames above, describe the object in the masked region in the video in detail.",
    "model_path": "models/DAM-3B-Video",
    "output_dir": "path/to/output",
    "normalized_coords": false,
    "no_stream": false
}
```

## CLI Options

### Input Options

- `--video_dir`: Directory containing video frames
- `--video_file`: Path to a video file
- `--points_list`: List of points for targeting objects, format: `[[[x1,y1], [x2,y2]], ...]`
- `--boxes_list`: List of boxes for targeting objects, format: `[[x1,y1,x2,y2], ...]`
- `--use_box`: Use bounding boxes instead of points

### Model Options

- `--model_path`: Path or HuggingFace ID of the DAM model (default: `nvidia/DAM-3B-Video`)
- `--prompt_mode`: Prompt mode (default: `focal_prompt`)
- `--conv_mode`: Conversation mode (default: `v1`)
- `--temperature`: Sampling temperature (default: `0.2`)
- `--top_p`: Top-p sampling parameter (default: `0.5`)

### Output Options

- `--output_image_path`: Path to save visualization images
- `--output_dir`: Directory to save outputs (visualizations and text)
- `--output_text_path`: Path to save the description text
- `--no_stream`: Disable streaming output

### Miscellaneous Options

- `--config`: Path to a configuration JSON file
- `--log_file`: Path to log file (default: `logs/processing.log`)
- `--normalized_coords`: Interpret coordinates as normalized (0-1) values

## Function Details

The DAM CLI wrapper consists of several Python modules that work together:

### Core Components

- **cli.py**: Command-line argument parsing
- **config.py**: Configuration handling
- **core.py**: Main processing logic
- **dam_utils.py**: DAM API utilities
- **download.py**: Model download utilities
- **logger.py**: Logging functionalities
- **processor.py**: Video and frame processing
- **sam_utils.py**: SAM integration for object segmentation
- **visualization.py**: Visualization utilities

### Scripts

- **dam_describe.py**: Main script for describing video regions
- **dam_download.py**: Utility for downloading models

## Troubleshooting

### Common Issues

1. **Model Download Errors**
   - Check your internet connection
   - Try downloading the model manually and placing it in the `models` directory

2. **CUDA/GPU Issues**
   - Ensure you have a compatible GPU
   - Install the CUDA-enabled version with `pip install -r requirements-cuda.txt`
   - Check that your GPU is recognized with `python -c "import torch; print(torch.cuda.is_available())"`

3. **Memory Errors**
   - The models require significant RAM/VRAM
   - Try using a smaller video or fewer frames

4. **Coordinate Parsing Errors**
   - Ensure the format for points or boxes is correct
   - Try using fewer points for better precision

### Logging

The tool generates logs in the `logs` directory. Check these logs for detailed information about any issues.

## License

This project maintains the original license from NVIDIA's Describe Anything Model (DAM) - Apache License 2.0.

## Acknowledgements

- [NVIDIA's Describe Anything Model (DAM)](https://github.com/NVlabs/describe-anything)
- [Meta's Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
