import argparse
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="NeSy Platform Tool")
    parser.add_argument("--check", action="store_true", help="Check system installation and prerequisites")
    args = parser.parse_args()

    if args.check:
        logger.info("Checking system installation...")
        missing_pkgs = False
        
        try:
            import nesy
            logger.info(f"✅ nesy package found: {nesy.__file__}")
        except ImportError:
            logger.error("❌ nesy package not found")
            missing_pkgs = True
            
        try:
            import torch
            logger.info(f"✅ torch found: {torch.__version__}")
        except ImportError:
            logger.error("❌ torch not found")
            missing_pkgs = True
            
        try:
            import cv2
            logger.info(f"✅ opencv-python found: {cv2.__version__}")
        except ImportError:
            logger.error("❌ opencv-python not found")
            missing_pkgs = True

        try:
            import transformers
            logger.info(f"✅ transformers found: {transformers.__version__}")
        except ImportError:
            logger.error("❌ transformers not found")
            missing_pkgs = True

        try:
            import ultralytics
            logger.info(f"✅ ultralytics found: {ultralytics.__version__}")
        except ImportError:
            logger.error("❌ ultralytics not found")
            missing_pkgs = True

        logger.info("System check complete.")
        
        if 'missing_pkgs' in locals() and missing_pkgs:
            logger.warning("\nSome dependencies are missing. Please install them:")
            logger.warning("pip install -r requirements.txt")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
