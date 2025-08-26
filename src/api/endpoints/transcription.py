"""
Transcription endpoints for the PS06 Speech-to-Text API.
"""

import time
import logging
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from ..models import (
    TranscriptionResponse, 
    TranscriptionSegment,
    BatchTranscriptionRequest,
    BatchTranscriptionResponse,
    ErrorResponse
)
from ..utils import (
    validate_audio_file, 
    save_uploaded_file, 
    get_audio_files_in_directory,
    create_temp_directory,
    cleanup_temp_directory,
    ensure_output_directory
)

# Import the transcriber from the main project
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from transcribe import WhisperTranscriber

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcription", tags=["transcription"])

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

@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: str = Form("auto", description="Language code (auto, en, hi, pa)"),
    device: str = Form("auto", description="Device to use (auto, cpu, cuda)")
):
    """
    Transcribe a single audio file.
    
    Upload an audio file and get the transcription with timestamps.
    """
    try:
        # Validate file
        validate_audio_file(file)
        
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
            
            # Convert segments to response format
            segments = []
            for segment in result.get('segments', []):
                segments.append(TranscriptionSegment(
                    text=segment.get('text', ''),
                    start_time=segment.get('start_time', 0.0),
                    end_time=segment.get('end_time', 0.0),
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
                device=result.get('device', device)
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

@router.post("/transcribe-batch", response_model=BatchTranscriptionResponse)
async def transcribe_batch(
    request: BatchTranscriptionRequest,
    background_tasks: BackgroundTasks
):
    """
    Transcribe multiple audio files in batch.
    
    Process all audio files in a directory and save transcripts.
    """
    try:
        # Get transcriber
        transcriber_instance = get_transcriber(request.device)
        
        # Validate input directory
        input_path = Path(request.input_dir)
        audio_files = get_audio_files_in_directory(input_path)
        
        if not audio_files:
            raise HTTPException(status_code=400, detail="No audio files found in input directory")
        
        # Ensure output directory
        output_path = Path(request.output_dir)
        ensure_output_directory(output_path)
        
        # Start batch transcription
        start_time = time.time()
        
        # Process files
        successful = 0
        failed = 0
        
        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {audio_file.name}")
                result = transcriber_instance.transcribe_audio(str(audio_file), request.language)
                
                # Save transcript
                output_file = output_path / f"{audio_file.stem}.json"
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                successful += 1
                logger.info(f"✓ {audio_file.name} transcribed successfully")
                
            except Exception as e:
                failed += 1
                logger.error(f"✗ Failed to transcribe {audio_file.name}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Create response
        response = BatchTranscriptionResponse(
            success=successful > 0,
            total_files=len(audio_files),
            successful_transcriptions=successful,
            failed_transcriptions=failed,
            success_rate=(successful / len(audio_files)) * 100 if len(audio_files) > 0 else 0,
            output_dir=str(output_path),
            processing_time=processing_time
        )
        
        logger.info(f"Batch transcription completed: {successful}/{len(audio_files)} successful")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")

@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for transcription.
    """
    try:
        transcriber_instance = get_transcriber()
        languages = transcriber_instance.get_supported_languages()
        
        return {
            "languages": languages,
            "total_languages": len(languages)
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(status_code=500, detail="Failed to get supported languages")

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    try:
        transcriber_instance = get_transcriber()
        model_loaded = transcriber_instance is not None
        
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "service": "PS06 Speech-to-Text API"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "error": str(e),
            "service": "PS06 Speech-to-Text API"
        }
