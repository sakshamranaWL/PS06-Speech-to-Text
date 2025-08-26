#!/usr/bin/env python3
"""
Test client for the PS06 Speech-to-Text API.
"""

import requests
import json
from pathlib import Path
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("\n🏠 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_supported_languages():
    """Test the supported languages endpoint."""
    print("\n🌍 Testing supported languages endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/transcription/supported-languages")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Supported languages failed: {e}")
        return False

def test_transcribe_audio(audio_file_path):
    """Test the transcribe endpoint with an audio file."""
    print(f"\n🎤 Testing transcribe endpoint with {audio_file_path}...")
    
    if not Path(audio_file_path).exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'language': 'auto',
                'device': 'auto'
            }
            
            response = requests.post(
                f"{BASE_URL}/transcription/transcribe",
                files=files,
                data=data
            )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Transcription successful!")
            print(f"Language: {result.get('language', 'Unknown')}")
            print(f"Text: {result.get('full_text', 'No text')}")
            print(f"Confidence: {result.get('confidence', 0):.3f}")
            print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        else:
            print(f"❌ Transcription failed: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

def test_batch_transcription():
    """Test the batch transcription endpoint."""
    print("\n📁 Testing batch transcription endpoint...")
    
    # Check if we have audio files in data/audio
    audio_dir = Path("data/audio")
    if not audio_dir.exists() or not list(audio_dir.glob("*")):
        print("❌ No audio files found in data/audio/")
        print("Please add some audio files to test batch transcription")
        return False
    
    try:
        data = {
            'input_dir': str(audio_dir),
            'output_dir': 'data/transcripts',
            'language': 'auto',
            'device': 'auto'
        }
        
        response = requests.post(
            f"{BASE_URL}/transcription/transcribe-batch",
            json=data
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch transcription successful!")
            print(f"Total files: {result.get('total_files', 0)}")
            print(f"Successful: {result.get('successful_transcriptions', 0)}")
            print(f"Failed: {result.get('failed_transcriptions', 0)}")
            print(f"Success rate: {result.get('success_rate', 0):.1f}%")
        else:
            print(f"❌ Batch transcription failed: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Batch transcription test failed: {e}")
        return False

def main():
    """Run all API tests."""
    print("🧪 PS06 Speech-to-Text API Test Suite")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    tests = [
        ("Health Check", test_health),
        ("Root Endpoint", test_root),
        ("Supported Languages", test_supported_languages),
        ("Batch Transcription", test_batch_transcription),
    ]
    
    # Check if we have test audio
    test_audio = Path("data/audio/test_audio.wav")
    if test_audio.exists():
        tests.append(("Single Audio Transcription", lambda: test_transcribe_audio(str(test_audio))))
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nTotal: {len(results)} | Passed: {passed} | Failed: {len(results) - passed}")
    
    if passed == len(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()

