"""
Main FastAPI application for PS06 Speech-to-Text API.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn
from pathlib import Path

from .endpoints.transcription import router as transcription_router
from .models import ErrorResponse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PS06 Speech-to-Text API...")
    
    # Ensure data directories exist
    data_dirs = ['data/audio', 'data/transcripts', 'data/uploads', 'logs']
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PS06 Speech-to-Text API...")

# Create FastAPI app
app = FastAPI(
    title="PS06 Speech-to-Text API",
    description="Multilingual speech recognition API using OpenAI Whisper model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

# Include routers
app.include_router(transcription_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PS06 Speech-to-Text API",
        "version": "1.0.0",
        "description": "Multilingual speech recognition using OpenAI Whisper",
        "endpoints": {
            "docs": "/docs",
            "health": "/transcription/health",
            "transcribe": "/transcription/transcribe",
            "transcribe-batch": "/transcription/transcribe-batch",
            "supported-languages": "/transcription/supported-languages"
        }
    }

# Health check endpoint
@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "healthy", "service": "PS06 Speech-to-Text API"}

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
