#!/usr/bin/env python3
"""
Startup script for PS06 Speech-to-Text API
"""

import uvicorn
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the FastAPI server."""
    try:
        logger.info("🚀 Starting PS06 Speech-to-Text API...")
        logger.info("📋 Model: OpenAI Whisper Large-v3")
        logger.info("🌐 Languages: English, Hindi, Punjabi")
        logger.info("🔧 Task: Transcribe (output in original language)")
        
        # Start the server
        uvicorn.run(
            "api:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info"
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
