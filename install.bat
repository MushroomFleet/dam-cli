@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements-cuda.txt

echo Installation complete!
echo To use CPU acceleration, run: pip install -r requirements-cpu.txt
echo.
echo See README.md for usage instructions.
