#!/usr/bin/env python3
"""
Command-line script for the DAM CLI
"""
import sys
import os

# Add parent directory to path to allow importing dam_cli
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dam_cli.cli import parse_arguments
from dam_cli.config import load_config, merge_config_with_args
from dam_cli.core import main
from dam_cli.logger import setup_logging, log_error

def run():
    """Run the DAM CLI tool"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging(args.log_file)
        
        # Load config if specified
        if args.config:
            try:
                config = load_config(args.config)
                args = merge_config_with_args(config, args)
            except Exception as e:
                log_error(logger, f"Error loading config file: {str(e)}")
                return 1
        
        # Run main function
        return main(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run())
