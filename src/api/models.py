"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class TranscriptionRequest(BaseModel):
    """Request model for transcription."""
    language: str = Field(default="auto", description="Language code (auto, en, hi, pa)")
    device: str = Field(default="auto", description="Device to use (auto, cpu, cuda)")


class TranscriptionSegment(BaseModel):
    """Model for transcription segment."""
    text: str = Field(description="Transcribed text")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    confidence: float = Field(description="Confidence score")


class TranscriptionResponse(BaseModel):
    """Response model for transcription."""
    success: bool = Field(description="Whether transcription was successful")
    audio_file: str = Field(description="Path to audio file")
    language: str = Field(description="Detected/used language")
    full_text: str = Field(description="Complete transcribed text")
    segments: List[TranscriptionSegment] = Field(description="Transcription segments with timestamps")
    confidence: float = Field(description="Overall confidence score")
    processing_time: float = Field(description="Processing time in seconds")
    model: str = Field(description="Model used for transcription")
    device: str = Field(description="Device used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of transcription")


class BatchTranscriptionRequest(BaseModel):
    """Request model for batch transcription."""
    input_dir: str = Field(description="Input directory containing audio files")
    output_dir: str = Field(description="Output directory for transcripts")
    language: str = Field(default="auto", description="Language code")
    device: str = Field(default="auto", description="Device to use")


class BatchTranscriptionResponse(BaseModel):
    """Response model for batch transcription."""
    success: bool = Field(description="Whether batch transcription was successful")
    total_files: int = Field(description="Total number of audio files")
    successful_transcriptions: int = Field(description="Number of successful transcriptions")
    failed_transcriptions: int = Field(description="Number of failed transcriptions")
    success_rate: float = Field(description="Success rate percentage")
    output_dir: str = Field(description="Output directory where transcripts were saved")
    processing_time: float = Field(description="Total processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of completion")


class SupportedLanguagesResponse(BaseModel):
    """Response model for supported languages."""
    languages: Dict[str, str] = Field(description="Language codes and names")
    total_languages: int = Field(description="Total number of supported languages")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Current timestamp")
    version: str = Field(description="API version")
    model_loaded: bool = Field(description="Whether the ASR model is loaded")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of error")
