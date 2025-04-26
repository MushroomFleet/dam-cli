#!/usr/bin/env python3
"""
Test script to verify that the installation is working correctly
"""
import os
import sys
import importlib

def test_imports():
    """Test that all necessary packages can be imported"""
    required_packages = [
        'torch',
        'transformers',
        'PIL',
        'cv2',
        'numpy',
        'tqdm',
        'huggingface_hub',
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    return len(missing_packages) == 0

def test_dam_cli():
    """Test that the dam_cli package can be imported"""
    try:
        import dam_cli
        print(f"✅ dam_cli is installed (version {dam_cli.__version__})")
        return True
    except ImportError:
        print("❌ dam_cli is not installed")
        return False

def test_scripts_exist():
    """Test that the scripts exist and are executable"""
    script_paths = [
        os.path.join('scripts', 'dam_describe.py'),
        os.path.join('scripts', 'dam_download.py'),
    ]
    
    missing_scripts = []
    
    for script_path in script_paths:
        if os.path.isfile(script_path):
            print(f"✅ {script_path} exists")
        else:
            missing_scripts.append(script_path)
            print(f"❌ {script_path} is missing")
    
    return len(missing_scripts) == 0

def main():
    """Run all tests"""
    print("Testing installation...")
    print("=" * 40)
    
    # Make sure we're in the correct directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.insert(0, project_root)
    
    # Run tests
    imports_ok = test_imports()
    dam_cli_ok = test_dam_cli()
    scripts_ok = test_scripts_exist()
    
    print("=" * 40)
    
    # Print overall result
    if imports_ok and dam_cli_ok and scripts_ok:
        print("✅ All tests passed! Installation seems working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
