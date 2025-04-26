"""
Main entry point for running the DAM CLI as a module
"""
import sys
from dam_cli.cli import parse_arguments
from dam_cli.core import main

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run main function and exit with its return code
    sys.exit(main(args))
