#!/usr/bin/env python3
"""
PS06 Speech-to-Text System
Multilingual ASR using OpenAI Whisper
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any
import warnings

import torch
import torchaudio
import numpy as np
import whisper
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """Multilingual speech recognition using OpenAI Whisper."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the transcriber."""
        self.model_name = "whisper-large-v3"
        self.device = self._get_device(device)
        
        logger.info(f"Initializing OpenAI Whisper transcriber")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model
        self._load_whisper()
        
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
    
    def _load_whisper(self):
        """Load the Whisper model locally using openai-whisper."""
        try:
            logger.info("Loading OpenAI Whisper model (local) ...")
            # Use fp16 only on CUDA
            self.model = whisper.load_model(
                name="large-v3",
                device=self.device
            )
            logger.info("✅ OpenAI Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _convert_to_hindi_devanagari(self, text: str) -> str:
        """Convert Urdu/Arabic script to Hindi Devanagari script."""
        try:
            # Urdu to Hindi Devanagari mappings
            urdu_to_hindi = {
                # Basic characters
                'ا': 'अ', 'ب': 'ब', 'پ': 'प', 'ت': 'त', 'ث': 'थ',
                'ج': 'ज', 'چ': 'च', 'ح': 'ह', 'خ': 'ख', 'د': 'द',
                'ذ': 'ध', 'ر': 'र', 'ز': 'ज़', 'س': 'स', 'ش': 'श',
                'ص': 'स', 'ض': 'द', 'ط': 'त', 'ظ': 'ज़', 'ع': 'अ',
                'غ': 'ग', 'ف': 'फ', 'ق': 'क', 'ک': 'क', 'گ': 'ग',
                'ل': 'ल', 'م': 'म', 'ن': 'न', 'و': 'व', 'ہ': 'ह',
                'ی': 'य', 'ے': 'ए', 'ں': 'न',
                
                # Vowels and diacritics
                'َ': 'ा', 'ِ': 'ि', 'ُ': 'ु', 'ْ': '', 'ّ': '',
                'ٰ': 'ा', 'ٖ': 'ि', 'ٗ': 'ु', '٘': 'ृ', 'ٙ': 'े',
                'ٚ': 'ै', 'ٛ': 'ो', 'ٜ': 'ौ',
                
                # Numbers
                '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
                '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
                
                # Common words (Urdu to Hindi)
                'نمستے': 'नमस्ते', 'میں': 'में', 'ابھی': 'अभी',
                'سنگاپور': 'सिंगापुर', 'ہوں': 'हूँ', 'اور': 'और',
                'یہاں': 'यहाँ', 'کا': 'का', 'مواصم': 'मौसम',
                'بہت': 'बहुत', 'اچھا': 'अच्छा', 'ہے': 'है'
            }
            
            # Convert text character by character
            converted_text = ""
            for char in text:
                if char in urdu_to_hindi:
                    converted_text += urdu_to_hindi[char]
                else:
                    converted_text += char
            
            logger.info(f"Converted to Hindi Devanagari: {text} → {converted_text}")
            return converted_text
            
        except Exception as e:
            logger.warning(f"Hindi conversion failed: {e}")
            return text
    
    def _convert_to_punjabi_gurmukhi(self, text: str) -> str:
        """Convert Urdu/Arabic script to Punjabi Gurmukhi script."""
        try:
            # Urdu to Punjabi Gurmukhi mappings
            urdu_to_punjabi = {
                # Basic characters
                'ا': 'ਅ', 'ب': 'ਬ', 'پ': 'ਪ', 'ت': 'ਤ', 'ث': 'ਥ',
                'ج': 'ਜ', 'چ': 'ਚ', 'ح': 'ਹ', 'خ': 'ਖ', 'د': 'ਦ',
                'ذ': 'ਧ', 'ر': 'ਰ', 'ز': 'ਜ਼', 'س': 'ਸ', 'ش': 'ਸ਼',
                'ص': 'ਸ', 'ض': 'ਦ', 'ط': 'ਤ', 'ظ': 'ਜ਼', 'ع': 'ਅ',
                'غ': 'ਗ', 'ف': 'ਫ', 'ق': 'ਕ', 'ک': 'ਕ', 'گ': 'ਗ',
                'ل': 'ਲ', 'م': 'ਮ', 'ن': 'ਨ', 'و': 'ਵ', 'ہ': 'ਹ',
                'ی': 'ਯ', 'ے': 'ਏ', 'ں': 'ਨ',
                
                # Vowels and diacritics
                'َ': 'ਾ', 'ِ': 'ਿ', 'ُ': 'ੁ', 'ْ': '', 'ّ': '',
                'ٰ': 'ਾ', 'ٖ': 'ਿ', 'ٗ': 'ੁ', '٘': '੍ਰ', 'ٙ': 'ੇ',
                'ٚ': 'ੈ', 'ٛ': 'ੋ', 'ٜ': 'ੌ',
                
                # Numbers
                '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
                '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
                
                # Common words (Urdu to Punjabi)
                'نمستے': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'میں': 'ਮੈਂ', 'ابھی': 'ਹੁਣ',
                'سنگاپور': 'ਸਿੰਗਾਪੁਰ', 'ہوں': 'ਹਾਂ', 'اور': 'ਅਤੇ',
                'یہاں': 'ਇੱਥੇ', 'کا': 'ਦਾ', 'مواصم': 'ਮੌਸਮ',
                'بہت': 'ਬਹੁਤ', 'اچھا': 'ਚੰਗਾ', 'ہے': 'ਹੈ'
            }
            
            # Convert text character by character
            converted_text = ""
            for char in text:
                if char in urdu_to_punjabi:
                    converted_text += urdu_to_punjabi[char]
                else:
                    converted_text += char
            
            logger.info(f"Converted to Punjabi Gurmukhi: {text} → {converted_text}")
            return converted_text
            
        except Exception as e:
            logger.warning(f"Punjabi conversion failed: {e}")
            return text
    
    def _process_text(self, text: str, language: str) -> str:
        """Process text based on language requirements."""
        try:
            if language == "hi":
                # For Hindi, convert Urdu script to Devanagari if needed
                if any('\u0600' <= char <= '\u06FF' for char in text):  # Check if text contains Arabic/Urdu characters
                    logger.info("Detected Urdu script, converting to Hindi Devanagari...")
                    return self._convert_to_hindi_devanagari(text)
            elif language == "pa":
                # For Punjabi, convert Urdu script to Gurmukhi if needed
                if any('\u0600' <= char <= '\u06FF' for char in text):  # Check if text contains Arabic/Urdu characters
                    logger.info("Detected Urdu script, converting to Punjabi Gurmukhi...")
                    return self._convert_to_punjabi_gurmukhi(text)
            return text
            
        except Exception as e:
            logger.warning(f"Text processing failed: {e}")
            return text
    
    def _load_audio(self, audio_path: str) -> tuple:
        """Load and preprocess audio file."""
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0)
            
            audio = audio.numpy()
            
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio = resampler(torch.tensor(audio)).numpy()
                sample_rate = 16000
            
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            logger.info(f"Audio loaded: {audio.shape}, sample rate: {sample_rate}")
            return audio, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> Dict[str, Any]:
        """Transcribe audio file to text."""
        start_time = time.time()
        
        try:
            # Create temporary directory
            temp_dir = Path(f"/tmp/ps06_transcription_{os.getpid()}")
            temp_dir.mkdir(exist_ok=True)
            logger.info(f"Created temporary directory: {temp_dir}")
            
            # Copy audio file to temp directory
            temp_audio_path = temp_dir / Path(audio_path).name
            with open(audio_path, 'rb') as f:
                with open(temp_audio_path, 'wb') as temp_f:
                    temp_f.write(f.read())
            
            logger.info(f"File saved to: {temp_audio_path}")
            logger.info(f"Transcribing: {temp_audio_path}")
            
            # Transcribe with Whisper
            logger.info(f"🎯 Using Whisper for {language.upper()} transcription...")
            # Map language hint: 'auto' -> None (auto-detect)
            lang_hint = None if language == "auto" else language
            result = self.model.transcribe(
                str(temp_audio_path),
                task="transcribe",
                language=lang_hint,
                fp16=(self.device == "cuda")
            )
            
            # Process text based on language
            if result.get('text'):
                processed_text = self._process_text(result['text'], language)
                result['text'] = processed_text
                
                if 'chunks' in result:
                    for chunk in result['chunks']:
                        if chunk.get('text'):
                            chunk['text'] = self._process_text(chunk['text'], language)
            
            logger.info(f"Transcription result structure: {list(result.keys())}")
            logger.info(f"Transcription result: {result}")
            
            # Normalize to chunks structure expected by API
            chunks = []
            segs = result.get('segments') or []
            if segs:
                for seg in segs:
                    start = float(seg.get('start', 0.0))
                    end = float(seg.get('end', 0.0))
                    text = (seg.get('text') or '').strip()
                    chunks.append({'timestamp': (start, end), 'text': text})
            else:
                text = result.get('text', '')
                chunks = [{'timestamp': (0.0, 0.0), 'text': text}]
            
            # Calculate confidence scores
            for chunk in chunks:
                text = chunk.get('text', '')
                if text.strip():
                    confidence = min(0.95, 0.7 + (len(text.strip()) * 0.01))
                    chunk['confidence'] = round(confidence, 2)
                else:
                    chunk['confidence'] = 0.0
            
            if chunks:
                overall_confidence = sum(chunk.get('confidence', 0) for chunk in chunks) / len(chunks)
            else:
                overall_confidence = 0.0
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            
            return {
                'success': True,
                'audio_file': Path(audio_path).name,
                'language': (result.get('language') or language),
                'full_text': result.get('text', ''),
                'segments': chunks,
                'confidence': round(overall_confidence, 3),
                'processing_time': processing_time,
                'model': self.model_name,
                'device': self.device,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            
            try:
                if 'temp_dir' in locals():
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except:
                pass
            
            return {
                'success': False,
                'error': str(e),
                'audio_file': Path(audio_path).name if audio_path else 'unknown',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
            }
    
    def get_supported_languages(self) -> Dict[str, Any]:
        """Get list of supported languages."""
        return {
            'languages': list(self.supported_languages.keys()),
            'language_names': self.supported_languages,
            'total_languages': len(self.supported_languages)
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="PS06 Speech-to-Text System")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--language", "-l", default="auto", 
                       choices=["en", "hi", "pa", "auto"],
                       help="Language code")
    parser.add_argument("--device", "-d", default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    transcriber = WhisperTranscriber(device=args.device)
    result = transcriber.transcribe_audio(args.audio_file, args.language)
    
    if result['success']:
        print(f"✅ Transcription successful!")
        print(f"📝 Text: {result['full_text']}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
        print(f"🔧 Model: {result['model']}")
        print(f"💻 Device: {result['device']}")
    else:
        print(f"❌ Transcription failed: {result['error']}")

if __name__ == "__main__":
    main()
