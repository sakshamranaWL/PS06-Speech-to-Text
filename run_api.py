#!/usr/bin/env python3
"""
Run the PS06 Speech-to-Text FastAPI server.
"""

import uvicorn
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    print("🚀 Starting PS06 Speech-to-Text API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Alternative docs at: http://localhost:8000/redoc")
    print("🏥 Health check at: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

