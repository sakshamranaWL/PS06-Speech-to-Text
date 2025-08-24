#!/usr/bin/env python3
"""
PS06 Speech-to-Text System
Multilingual ASR using OpenAI Whisper model
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

import torch
import torchaudio
import librosa
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Multilingual speech recognition using OpenAI Whisper model."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the transcriber.
        
        Args:
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.model_name = "openai/whisper-base"
        self.device = self._get_device(device)
        
        logger.info(f"Initializing OpenAI Whisper transcriber")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        self._load_model()
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'pa': 'Punjabi'
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the ASR model and processor."""
        try:
            logger.info("Loading OpenAI Whisper model and processor...")
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info("✅ OpenAI Whisper model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code ('en', 'hi', 'pa') or 'auto' for detection
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Transcribing: {audio_path}")
        
        # Validate language
        if language != "auto" and language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.supported_languages.keys())}")
        
        # Load and preprocess audio
        audio, sample_rate = self._load_audio(audio_path)
        
        # Transcribe
        start_time = time.time()
        
        try:
            # Run transcription
            result = self.pipeline(
                audio,
                sampling_rate=sample_rate,
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5
            )
            
            processing_time = time.time() - start_time
            
            # Process results
            transcription = self._process_transcription_result(result, language)
            
            # Add metadata
            transcription.update({
                'audio_file': audio_path,
                'processing_time': processing_time,
                'model': self.model_name,
                'device': self.device
            })
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            return transcription
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _load_audio(self, audio_path: str) -> tuple:
        """Load and preprocess audio file."""
        try:
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            logger.info(f"Audio loaded: {audio.shape}, sample rate: {sample_rate}")
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def _process_transcription_result(self, result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Process the transcription result."""
        transcription = {
            'language': language,
            'segments': [],
            'full_text': result.get('text', ''),
            'confidence': result.get('confidence', 0.0)
        }
        
        # Process timestamps if available
        if 'chunks' in result:
            for chunk in result['chunks']:
                segment = {
                    'text': chunk.get('text', ''),
                    'start_time': chunk.get('timestamp', [0, 0])[0],
                    'end_time': chunk.get('timestamp', [0, 0])[1],
                    'confidence': chunk.get('confidence', 0.0)
                }
                transcription['segments'].append(segment)
        
        # If no chunks, create single segment
        if not transcription['segments']:
            transcription['segments'] = [{
                'text': transcription['full_text'],
                'start_time': 0.0,
                'end_time': 0.0,
                'confidence': transcription['confidence']
            }]
        
        return transcription
    
    def transcribe_batch(self, input_dir: str, output_dir: str, language: str = "auto") -> List[Dict[str, Any]]:
        """Transcribe multiple audio files in batch.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to save transcripts
            language: Language code or 'auto'
            
        Returns:
            List of transcription results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(input_path.glob(f'*{ext}'))
            audio_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        # Process files
        results = []
        successful = 0
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing {i}/{len(audio_files)}: {audio_file.name}")
            
            try:
                # Transcribe
                result = self.transcribe_audio(str(audio_file), language)
                results.append(result)
                
                # Save transcript
                output_file = output_path / f"{audio_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                successful += 1
                logger.info(f"✓ {audio_file.name} transcribed successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to transcribe {audio_file.name}: {e}")
                continue
        
        # Save batch summary
        summary = {
            'total_files': len(audio_files),
            'successful_transcriptions': successful,
            'failed_transcriptions': len(audio_files) - successful,
            'success_rate': successful / len(audio_files) if len(audio_files) > 0 else 0,
            'model': self.model_name,
            'language': language
        }
        
        summary_file = output_path / "transcription_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch transcription completed: {successful}/{len(audio_files)} successful")
        logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages."""
        return self.supported_languages.copy()

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="OpenAI Whisper Speech-to-Text Transcription")
    
    parser.add_argument(
        '--audio', '-a',
        type=str,
        help='Path to single audio file'
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Input directory for batch processing'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='data/transcripts',
        help='Output directory for transcripts'
    )
    
    parser.add_argument(
        '--language', '-l',
        type=str,
        default='auto',
        choices=['auto', 'en', 'hi', 'pa'],
        help='Language code (auto, en, hi, pa)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.audio and not args.input_dir:
        parser.error("Either --audio or --input-dir must be specified")
    
    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber(device=args.device)
        
        # Show supported languages
        languages = transcriber.get_supported_languages()
        logger.info(f"Supported languages: {languages}")
        
        if args.audio:
            # Single file transcription
            result = transcriber.transcribe_audio(args.audio, args.language)
            
            # Save result
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            output_file = output_path / f"{Path(args.audio).stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription saved to: {output_file}")
            
            # Print result
            print(f"\n🎤 Transcription Result:")
            print(f"Language: {result['language']}")
            print(f"Text: {result['full_text']}")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Processing time: {result['processing_time']:.2f}s")
            
        elif args.input_dir:
            # Batch transcription
            results = transcriber.transcribe_batch(
                args.input_dir,
                args.output_dir,
                args.language
            )
            
            logger.info(f"Batch transcription completed. {len(results)} results saved to {args.output_dir}")
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
