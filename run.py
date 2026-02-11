#!/usr/bin/env python3
"""
Automated Fine-tuning System - Main Entry Point

Usage:
    # Install packages first
    python run.py --install-packages
    
    # Run experiments after installation
    python run.py config.json
    python run.py config1.json config2.json --threshold-f1 0.3
"""

import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('finetuning_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import argparse
import argparse


def load_env_file(env_path=".env"):
    """Load environment variables from a .env file if they are not already set."""
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def install_packages_command():
    """Install required packages with GPU detection."""
    logger.info("=" * 60)
    logger.info("Starting package installation with GPU detection...")
    logger.info("=" * 60)
    try:
        from src.utils import install_packages
        install_packages()
        logger.info("=" * 60)
        logger.info("Package installation completed successfully!")
        logger.info("You can now run experiments using: python run.py config.json")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error("CRITICAL: Package installation failed!")
        logger.error("=" * 60)
        logger.error(f"Error: {str(e)}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Check internet connection")
        logger.error("2. Verify GPU drivers: nvidia-smi")
        logger.error("3. Check logs for detailed error messages")
        logger.error("=" * 60)
        sys.exit(1)


def main():
    load_env_file()
    parser = argparse.ArgumentParser(
        description='Automated Fine-tuning System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install packages first
  python run.py --install-packages
  
  # Run experiments after installation
  python run.py config.json
  python run.py config1.json config2.json --threshold-f1 0.3
        """
    )
    parser.add_argument('--install-packages', action='store_true', 
                       help='Install required packages with GPU detection')
    parser.add_argument('config_files', nargs='*', help='Path(s) to config JSON file(s)')
    parser.add_argument('--threshold-f1', type=float, default=0.2, help='F1 threshold (default: 0.2)')
    parser.add_argument('--threshold-latency', type=float, default=0.3, help='Latency threshold (default: 0.3)')
    parser.add_argument('--factors', nargs='+', default=['accuracy', 'latency'], help='Optimization factors (default: accuracy latency)')
    
    args = parser.parse_args()
    
    # Handle package installation command
    if args.install_packages:
        install_packages_command()
        return
    
    # Check if config files are provided
    if not args.config_files:
        parser.error("No config files provided. Use --install-packages to install packages first, or provide config file(s).")
    
    # Check if torch is available (packages are installed)
    try:
        import torch
    except ImportError:
        logger.error("=" * 60)
        logger.error("ERROR: Required packages are not installed!")
        logger.error("=" * 60)
        logger.error("Please run package installation first:")
        logger.error("  python run.py --install-packages")
        logger.error("=" * 60)
        sys.exit(1)
    
    # Import engine after confirming packages are installed
    from src.engine import RecommendationEngine
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("No GPU detected. Training will be slow on CPU.")
    
    logger.info(f"Config files: {', '.join(args.config_files)}")
    
    # Run experiments
    recommendation = RecommendationEngine(
        config_paths=args.config_files,
        threshold_f1=args.threshold_f1,
        threshold_latency=args.threshold_latency,
        factors=args.factors
    )
    
    result = recommendation.get_best_model()
    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(result)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
