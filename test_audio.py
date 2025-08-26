#!/usr/bin/env python3
"""Test script to debug audio loading issues"""

import torchaudio
import soundfile as sf
import librosa
import numpy as np

def test_torchaudio():
    print("Testing torchaudio...")
    try:
        audio, sr = torchaudio.load("data/audio/test_audio.wav")
        print(f"✅ Torchaudio: {audio.shape}, {sr}Hz")
        return True
    except Exception as e:
        print(f"❌ Torchaudio failed: {e}")
        return False

def test_soundfile():
    print("Testing soundfile...")
    try:
        audio, sr = sf.read("data/audio/test_audio.wav")
        print(f"✅ Soundfile: {audio.shape}, {sr}Hz")
        return True
    except Exception as e:
        print(f"❌ Soundfile failed: {e}")
        return False

def test_librosa():
    print("Testing librosa...")
    try:
        audio, sr = librosa.load("data/audio/test_audio.wav", sr=16000)
        print(f"✅ Librosa: {audio.shape}, {sr}Hz")
        return True
    except Exception as e:
        print(f"❌ Librosa failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing audio loading methods...")
    print("=" * 50)
    
    test_torchaudio()
    test_soundfile()
    test_librosa()
