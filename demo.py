#!/usr/bin/env python3
"""
Demo script for PS06 Speech-to-Text System
Shows how to use the AI4Bharat IndicConformer model
"""

import os
import sys
from pathlib import Path

def show_banner():
    """Show system banner."""
    print("🎤" + "="*60 + "🎤")
    print("🚀 PS06 Speech-to-Text System")
    print("🌍 AI4Bharat IndicConformer Model")
    print("🎯 Hindi • English • Punjabi")
    print("🎤" + "="*60 + "🎤")

def show_features():
    """Show system features."""
    print("\n🌟 Key Features:")
    print("   ✅ Multilingual ASR (Hindi, English, Punjabi)")
    print("   ✅ AI4Bharat IndicConformer model")
    print("   ✅ Audio format support (WAV, MP3, M4A, FLAC)")
    print("   ✅ Batch processing capability")
    print("   ✅ Confidence scoring")
    print("   ✅ Timestamp generation")

def show_usage():
    """Show usage examples."""
    print("\n📖 Usage Examples:")
    print("\n1. Single file transcription:")
    print("   python transcribe.py --audio audio.wav --language hi")
    print("   python transcribe.py --audio audio.wav --language en")
    print("   python transcribe.py --audio audio.wav --language pa")
    
    print("\n2. Batch processing:")
    print("   python transcribe.py --input-dir data/audio --output-dir data/transcripts")
    
    print("\n3. Language-specific batch:")
    print("   python transcribe.py --input-dir hindi_audio/ --language hi")
    print("   python transcribe.py --input-dir english_audio/ --language en")
    print("   python transcribe.py --input-dir punjabi_audio/ --language pa")

def show_commands():
    """Show available make commands."""
    print("\n🛠️  Available Commands:")
    print("   make install              # Install dependencies")
    print("   make test                 # Run system tests")
    print("   make download-model       # Download AI4Bharat model")
    print("   make verify-model         # Verify downloaded model")
    print("   make model-info           # Show model information")
    print("   make transcribe-help      # Show transcription help")

def show_languages():
    """Show supported languages."""
    print("\n🌍 Supported Languages:")
    print("   🇮🇳 Hindi (hi) - Devanagari script")
    print("   🇺🇸 English (en) - Latin script")
    print("   🇮🇳 Punjabi (pa) - Gurmukhi script")

def show_output_format():
    """Show output format."""
    print("\n📊 Output Format:")
    print("   • JSON transcripts with metadata")
    print("   • Per-segment confidence scores")
    print("   • Timestamp information")
    print("   • Language identification")
    print("   • Processing statistics")

def show_quick_start():
    """Show quick start steps."""
    print("\n🚀 Quick Start:")
    print("   1. Install dependencies: make install")
    print("   2. Test system: make test")
    print("   3. Download model: make download-model")
    print("   4. Verify model: make verify-model")
    print("   5. Start transcribing!")

def main():
    """Main demo function."""
    show_banner()
    
    show_features()
    show_languages()
    show_output_format()
    show_usage()
    show_commands()
    show_quick_start()
    
    print("\n🎉 Ready to transcribe! Check the README.md for detailed instructions.")
    print("🎤" + "="*60 + "🎤")

if __name__ == "__main__":
    main()
