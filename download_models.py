#!/usr/bin/env python3
"""
Model Download Utility for PS06 Speech-to-Text System
Downloads AI4Bharat IndicConformer model from AI Kosh
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Download AI4Bharat IndicConformer model."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize the model downloader.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Single model - AI4Bharat IndicConformer
        self.model_info = {
            'name': 'AI4Bharat IndicConformer 600M Multilingual',
            'description': '600M parameter multilingual ASR model for Indian languages (Hindi, English, Punjabi)',
            'languages': ['hi', 'en', 'pa'],
            'source': 'ai4bharat',
            'download_url': 'https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual',
            'huggingface_id': 'ai4bharat/indic-conformer-600m-multilingual',
            'model_id': 'indic-conformer-600m-multilingual'
        }
    
    def show_model_info(self) -> None:
        """Show information about the model."""
        print("\n🚀 AI4Bharat IndicConformer Model")
        print("=" * 60)
        print(f"📦 Name: {self.model_info['name']}")
        print(f"📝 Description: {self.model_info['description']}")
        print(f"🌍 Languages: {', '.join(self.model_info['languages'])}")
        print(f"🔗 Source: {self.model_info['source']}")
        print(f"📥 Download: {self.model_info['download_url']}")
        
        # Check if already downloaded
        model_path = self.models_dir / self.model_info['model_id']
        if model_path.exists():
            print(f"✅ Status: Already downloaded to {model_path}")
        else:
            print(f"⏳ Status: Not downloaded")
        
        print("=" * 60)
    
    def download_model(self, force: bool = False) -> bool:
        """Download the AI4Bharat IndicConformer model.
        
        Args:
            force: Force re-download if already exists
            
        Returns:
            True if successful, False otherwise
        """
        model_path = self.models_dir / self.model_info['model_id']
        
        # Check if already downloaded
        if model_path.exists() and not force:
            logger.info(f"Model already exists at {model_path}")
            logger.info("Use --force to re-download")
            return True
        
        logger.info(f"Downloading {self.model_info['name']}...")
        logger.info(f"Languages: {', '.join(self.model_info['languages'])}")
        logger.info("This may take several minutes depending on your internet connection...")
        
        try:
            # Download using transformers
            from transformers import AutoProcessor, AutoModelForCTC
            
            logger.info("Downloading model and processor...")
            
            # Download processor
            processor = AutoProcessor.from_pretrained(
                self.model_info['huggingface_id'],
                cache_dir=str(model_path)
            )
            
            # Download model
            model = AutoModelForCTC.from_pretrained(
                self.model_info['huggingface_id'],
                cache_dir=str(model_path)
            )
            
            # Save model info
            model_info_file = model_path / "model_info.json"
            with open(model_info_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_id': self.model_info['model_id'],
                    'name': self.model_info['name'],
                    'description': self.model_info['description'],
                    'languages': self.model_info['languages'],
                    'source': self.model_info['source'],
                    'huggingface_id': self.model_info['huggingface_id'],
                    'download_date': str(Path().cwd()),
                    'model_size_mb': self._get_model_size(model_path)
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Model downloaded successfully!")
            logger.info(f"Location: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False
    
    def _get_model_size(self, model_path: Path) -> float:
        """Get model size in MB."""
        try:
            total_size = 0
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return round(total_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def verify_model(self) -> bool:
        """Verify that the downloaded model is working."""
        model_path = self.models_dir / self.model_info['model_id']
        
        if not model_path.exists():
            logger.error(f"Model not found at {model_path}")
            return False
        
        try:
            # Try to load the model
            from transformers import AutoProcessor, AutoModelForCTC
            
            logger.info("Verifying model...")
            
            processor = AutoProcessor.from_pretrained(str(model_path))
            model = AutoModelForCTC.from_pretrained(str(model_path))
            
            logger.info("✅ Model loads successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model verification failed: {e}")
            return False
    
    def get_model_path(self) -> str:
        """Get the path to the downloaded model."""
        return str(self.models_dir / self.model_info['model_id'])

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Download AI4Bharat IndicConformer Model")
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show model information'
    )
    
    parser.add_argument(
        '--download', '-d',
        action='store_true',
        help='Download the model'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download if model already exists'
    )
    
    parser.add_argument(
        '--verify', '-v',
        action='store_true',
        help='Verify downloaded model'
    )
    
    parser.add_argument(
        '--models-dir', '-m',
        type=str,
        default='models',
        help='Directory to store models'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = ModelDownloader(args.models_dir)
    
    try:
        if args.info:
            # Show model info
            downloader.show_model_info()
            
        elif args.download:
            # Download model
            success = downloader.download_model(args.force)
            if success:
                logger.info("Model downloaded successfully!")
                logger.info(f"Model location: {downloader.get_model_path()}")
            else:
                logger.error("Failed to download model")
                return 1
                
        elif args.verify:
            # Verify model
            success = downloader.verify_model()
            if success:
                logger.info("Model verification successful!")
            else:
                logger.error("Model verification failed")
                return 1
                
        else:
            # Show help if no action specified
            downloader.show_model_info()
            print("\nUsage:")
            print("  python download_models.py --info      # Show model information")
            print("  python download_models.py --download # Download the model")
            print("  python download_models.py --verify    # Verify downloaded model")
            
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
