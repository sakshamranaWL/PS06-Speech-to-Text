#!/usr/bin/env python3
"""
Test script for PS06 Speech-to-Text API
"""

import requests
import json
import time
from pathlib import Path

# API configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint."""
    print("\n🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"API Name: {data.get('name')}")
        print(f"Version: {data.get('version')}")
        print(f"Supported Languages: {data.get('supported_languages')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_supported_languages():
    """Test the supported languages endpoint."""
    print("\n🔍 Testing supported languages endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/supported-languages")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Languages: {data.get('languages')}")
        print(f"Total Languages: {data.get('total_languages')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Supported languages failed: {e}")
        return False

def test_transcription(audio_file_path: str, language: str = "auto"):
    """Test transcription with a sample audio file."""
    print(f"\n🎵 Testing transcription with {audio_file_path} (language: {language})...")
    
    if not Path(audio_file_path).exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': (Path(audio_file_path).name, f, 'audio/wav')}
            data = {'language': language, 'device': 'auto'}
            
            start_time = time.time()
            response = requests.post(f"{API_BASE_URL}/transcribe", files=files, data=data)
            processing_time = time.time() - start_time
            
            print(f"Status: {response.status_code}")
            print(f"Request processing time: {processing_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transcription successful!")
                print(f"📝 Full text: {result.get('full_text', '')}")
                print(f"🎯 Language: {result.get('language', '')}")
                print(f"📊 Confidence: {result.get('confidence', 0)}")
                print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"🔧 Model: {result.get('model', '')}")
                print(f"💻 Device: {result.get('device', '')}")
                return True
            else:
                print(f"❌ Transcription failed: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting PS06 Speech-to-Text API Tests")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_check()
    root_ok = test_root_endpoint()
    languages_ok = test_supported_languages()
    
    # Test transcription with sample files
    audio_files = [
        ("data/audio/test_audio.wav", "auto"),
        ("data/audio/english1.mp3", "en"),
        ("data/audio/hindi1.mp3", "hi"),
        ("data/audio/punjabi1.mp3", "pa")
    ]
    
    transcription_results = []
    for audio_file, language in audio_files:
        if Path(audio_file).exists():
            result = test_transcription(audio_file, language)
            transcription_results.append(result)
        else:
            print(f"⚠️  Skipping {audio_file} - file not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Root Endpoint: {'✅ PASS' if root_ok else '❌ FAIL'}")
    print(f"Supported Languages: {'✅ PASS' if languages_ok else '❌ FAIL'}")
    print(f"Transcription Tests: {sum(transcription_results)}/{len(transcription_results)} PASS")
    
    if all([health_ok, root_ok, languages_ok]) and transcription_results:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the API configuration.")

if __name__ == "__main__":
    main()
