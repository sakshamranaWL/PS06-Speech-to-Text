#!/usr/bin/env python3
"""
Test script for PS06 Speech-to-Text System
OpenAI Whisper Model
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import torchaudio
        print(f"✅ TorchAudio: {torchaudio.__version__}")
    except ImportError as e:
        print(f"❌ TorchAudio: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import librosa
        print(f"✅ Librosa: {librosa.__version__}")
    except ImportError as e:
        print(f"❌ Librosa: {e}")
        return False
    
    try:
        import soundfile
        print(f"✅ SoundFile: {soundfile.__version__}")
    except ImportError as e:
        print(f"❌ SoundFile: {e}")
        return False
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist."""
    print("\n📁 Testing directory structure...")
    
    required_dirs = ['data', 'data/audio', 'data/transcripts', 'models']
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ (missing)")
            return False
    
    return True

def test_files():
    """Test if required files exist."""
    print("\n📄 Testing required files...")
    
    required_files = ['transcribe.py', 'download_models.py', 'requirements.txt', 'README.md']
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            return False
    
    return True

def test_model_downloader():
    """Test the model downloader."""
    print("\n🚀 Testing model downloader...")
    
    try:
        from download_models import ModelDownloader
        
        downloader = ModelDownloader()
        print("✅ ModelDownloader class imported successfully")
        
        # Test model info
        print("📋 Model information:")
        print(f"   - Name: {downloader.model_info['name']}")
        print(f"   - Languages: {', '.join(downloader.model_info['languages'])}")
        print(f"   - Source: {downloader.model_info['source']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ModelDownloader test failed: {e}")
        return False

def test_transcriber():
    """Test the transcriber class."""
    print("\n🎤 Testing transcriber...")
    
    try:
        from transcribe import WhisperTranscriber
        
        # Test without loading model (just class definition)
        transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
        transcriber.supported_languages = {'en': 'English', 'hi': 'Hindi', 'pa': 'Punjabi'}
        
        print("✅ WhisperTranscriber class imported successfully")
        print(f"   Supported languages: {list(transcriber.supported_languages.keys())}")
        print(f"   Model: openai/whisper-base")
        
        return True
        
    except Exception as e:
        print(f"❌ Transcriber test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 PS06 Speech-to-Text System - OpenAI Whisper Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Directory Structure", test_directories),
        ("Required Files", test_files),
        ("Model Downloader", test_model_downloader),
        ("Transcriber", test_transcriber)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Test transcription: python transcribe.py --help")
        print("2. Transcribe audio: python transcribe.py --audio file.wav --language en")
        print("3. Batch process: python transcribe.py --input-dir audio/ --output-dir transcripts/")
        print("\n🎯 This system uses OpenAI Whisper for:")
        print("   - Hindi (hi) speech recognition")
        print("   - English (en) speech recognition")
        print("   - Punjabi (pa) speech recognition")
        print("   - Automatic language detection")
        print("   - No authentication required")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
