#!/usr/bin/env python3
"""
PS06 Speech-to-Text API
"""

import time
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our transcriber
from transcribe import WhisperTranscriber

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="PS06 Speech-to-Text API",
    description="Multilingual speech recognition API supporting English, Hindi, and Punjabi",
    version="1.0.0"
)

# Pydantic models
class TranscriptionSegment(BaseModel):
    text: str
    start_time: float
    end_time: float
    confidence: float

class TranscriptionResponse(BaseModel):
    success: bool
    audio_file: str
    language: str
    full_text: str
    segments: List[TranscriptionSegment]
    confidence: float
    processing_time: float
    model: str
    device: str
    timestamp: str

class BatchTranscriptionResult(BaseModel):
    filename: str
    success: bool
    transcription: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchTranscriptionResponse(BaseModel):
    success: bool
    total_files: int
    successful_transcriptions: int
    failed_transcriptions: int
    success_rate: float
    processing_time: float
    results: List[BatchTranscriptionResult]
    model: str
    device: str
    timestamp: str

class ErrorResponse(BaseModel):
    error: str
    detail: str

# Global transcriber instance
transcriber = None

def get_transcriber(device: str = "auto") -> WhisperTranscriber:
    """Get or create transcriber instance."""
    global transcriber
    if transcriber is None:
        try:
            transcriber = WhisperTranscriber(device=device)
            logger.info("Transcriber initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize transcription service")
    return transcriber

def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Check file extension
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format: {file_ext}. Supported formats: {allowed_extensions}"
        )
    
    # Check file size (max 50MB)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size: 50MB")

async def save_uploaded_file(file: UploadFile, file_path: Path) -> None:
    """Save uploaded file to disk."""
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

def create_temp_directory() -> Path:
    """Create temporary directory for processing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ps06_transcription_"))
    logger.info(f"Created temporary directory: {temp_dir}")
    return temp_dir

def cleanup_temp_directory(temp_dir: Path) -> None:
    """Clean up temporary directory."""
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

# API Endpoints
@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "name": "PS06 Speech-to-Text API",
        "version": "1.0.0",
        "description": "Multilingual speech recognition API",
        "endpoints": {
            "transcribe": "/transcribe",
            "batch_transcribe": "/transcribe-batch", 
            "languages": "/supported-languages",
            "health": "/health",
            "docs": "/docs"
        },
        "supported_languages": ["en", "hi", "pa"],
        "models": {
            "en": "OpenAI Whisper",
            "hi": "OpenAI Whisper + Hindi Devanagari Converter",
            "pa": "OpenAI Whisper + Punjabi Gurmukhi Converter"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        transcriber_instance = get_transcriber()
        model_loaded = transcriber_instance is not None
        
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "service": "PS06 Speech-to-Text API",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
            "service": "PS06 Speech-to-Text API",
            "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S')
        }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("auto", description="Language code (auto, en, hi, pa)"),
    device: str = Form("auto", description="Device to use (auto, cpu, cuda)")
):
    """
    Transcribe a single audio file.
    
    **Language Options:**
    - `auto`: Auto-detect language
    - `en`: English
    - `hi`: Hindi (outputs in Devanagari script)
    - `pa`: Punjabi (outputs in Gurmukhi script)
    
    **Supported Formats:** WAV, MP3, M4A, FLAC, OGG, AAC (max 50MB)
    """
    try:
        # Validate file
        validate_audio_file(file)
        
        # Validate language selection
        valid_languages = ['auto', 'en', 'hi', 'pa']
        if language not in valid_languages:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid language '{language}'. Supported languages: {valid_languages}"
            )
        
        # Get transcriber
        transcriber_instance = get_transcriber(device)
        
        # Create temporary directory for processing
        temp_dir = create_temp_directory()
        
        try:
            # Save uploaded file
            temp_audio_path = temp_dir / file.filename
            await save_uploaded_file(file, temp_audio_path)
            
            # Transcribe
            start_time = time.time()
            result = transcriber_instance.transcribe_audio(str(temp_audio_path), language)
            processing_time = time.time() - start_time
            
            if not result.get('success', False):
                raise HTTPException(status_code=500, detail=result.get('error', 'Transcription failed'))
            
            # Convert segments to response format
            segments = []
            for segment in result.get('segments', []):
                segments.append(TranscriptionSegment(
                    text=segment.get('text', ''),
                    start_time=segment.get('timestamp', (0.0, 0.0))[0],
                    end_time=segment.get('timestamp', (0.0, 0.0))[1],
                    confidence=segment.get('confidence', 0.0)
                ))
            
            # Create response
            response = TranscriptionResponse(
                success=True,
                audio_file=file.filename,
                language=result.get('language', language),
                full_text=result.get('full_text', ''),
                segments=segments,
                confidence=result.get('confidence', 0.0),
                processing_time=processing_time,
                model=result.get('model', ''),
                device=result.get('device', device),
                timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')
            )
            
            logger.info(f"Successfully transcribed {file.filename}")
            return response
            
        finally:
            # Clean up temporary files
            cleanup_temp_directory(temp_dir)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe-batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files to transcribe"),
    language: str = Form("auto", description="Language code: 'en', 'hi', 'pa', or 'auto'"),
    device: str = Form("auto", description="Device to use: 'cpu', 'cuda', or 'auto'")
):
    """
    Transcribe multiple audio files in batch.
    
    **Language Options:**
    - `auto`: Auto-detect language
    - `en`: English
    - `hi`: Hindi (outputs in Devanagari script)
    - `pa`: Punjabi (outputs in Gurmukhi script)
    
    **Supported Formats:** WAV, MP3, M4A, FLAC, OGG, AAC (max 50MB per file)
    """
    try:
        # Get transcriber
        transcriber_instance = get_transcriber(device)
        
        # Validate files
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Start batch transcription
        start_time = time.time()
        
        # Process files
        successful = 0
        failed = 0
        results = []
        
        for file in files:
            try:
                logger.info(f"Processing: {file.filename}")
                
                # Validate file
                validate_audio_file(file)
                
                # Create temporary directory for this file
                temp_dir = create_temp_directory()
                
                try:
                    # Save uploaded file
                    temp_audio_path = temp_dir / file.filename
                    await save_uploaded_file(file, temp_audio_path)
                    
                    # Transcribe
                    result = transcriber_instance.transcribe_audio(str(temp_audio_path), language)
                    
                    if result.get('success', False):
                        successful += 1
                        results.append(BatchTranscriptionResult(
                            filename=file.filename,
                            success=True,
                            transcription=result
                        ))
                        logger.info(f"✓ {file.filename} transcribed successfully")
                    else:
                        failed += 1
                        results.append(BatchTranscriptionResult(
                            filename=file.filename,
                            success=False,
                            error=result.get('error', 'Unknown error')
                        ))
                        logger.warning(f"✗ {file.filename} transcription failed")
                        
                finally:
                    # Clean up temporary files for this file
                    cleanup_temp_directory(temp_dir)
                    
            except Exception as e:
                failed += 1
                results.append(BatchTranscriptionResult(
                    filename=file.filename,
                    success=False,
                    error=str(e)
                ))
                logger.error(f"✗ Failed to process {file.filename}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Create response
        response = BatchTranscriptionResponse(
            success=True,
            total_files=len(files),
            successful_transcriptions=successful,
            failed_transcriptions=failed,
            success_rate=successful / len(files) if len(files) > 0 else 0,
            processing_time=processing_time,
            results=results,
            model=transcriber_instance.model_name,
            device=device,
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')
        )
        
        logger.info(f"Batch transcription completed: {successful}/{len(files)} successful")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")

@app.get("/supported-languages")
async def get_supported_languages():
    """Get supported languages and model information."""
    return {
        "languages": {
            "en": "English",
            "hi": "Hindi", 
            "pa": "Punjabi"
        },
        "total_languages": 3,
        "recommendation": "Choose specific language (en/hi/pa) instead of 'auto' for better accuracy",
        "examples": {
            "en": "Use for English speech like 'Hello, how are you?'",
            "hi": "Use for Hindi speech like 'नमस्ते, कैसे हो आप?' (outputs in Devanagari)",
            "pa": "Use for Punjabi speech like 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਕਿਵੇਂ ਹੋ ਤੁਸੀਂ?' (outputs in Gurmukhi)"
        },
        "model_info": {
            "en": "OpenAI Whisper",
            "hi": "OpenAI Whisper + Hindi Devanagari Converter",
            "pa": "OpenAI Whisper + Punjabi Gurmukhi Converter"
        },
        "benefits": {
            "en": "High accuracy for English speech",
            "hi": "Native Hindi transcription in Devanagari script",
            "pa": "Native Punjabi transcription in Gurmukhi script"
        },
        "supported_formats": ["WAV", "MP3", "M4A", "FLAC", "OGG", "AAC"],
        "max_file_size": "50MB"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
