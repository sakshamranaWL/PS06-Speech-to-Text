"""
Utility functions for the API.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple
import aiofiles
from fastapi import UploadFile, HTTPException
import logging

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}

def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported audio format. Supported formats: {', '.join(SUPPORTED_AUDIO_FORMATS)}"
        )
    
    # Check file size (limit to 100MB)
    if file.size and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size too large. Maximum size: 100MB")

async def save_uploaded_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(destination, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        logger.info(f"File saved to: {destination}")
        return destination
        
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

def get_audio_files_in_directory(directory: Path) -> List[Path]:
    """Get all audio files in a directory."""
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {directory}")
    
    audio_files = []
    for ext in SUPPORTED_AUDIO_FORMATS:
        audio_files.extend(directory.glob(f"*{ext}"))
        audio_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(audio_files)

def create_temp_directory() -> Path:
    """Create a temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ps06_transcription_"))
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_directory(temp_dir: Path) -> None:
    """Clean up temporary directory."""
    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

def ensure_output_directory(output_dir: Path) -> Path:
    """Ensure output directory exists."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0
