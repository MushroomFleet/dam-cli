#!/bin/bash
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installation complete!"
echo "To use GPU acceleration, run: pip install -r requirements-cuda.txt"
echo ""
echo "See README.md for usage instructions."
