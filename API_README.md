# PS06 Speech-to-Text FastAPI

A modern REST API for the PS06 Speech-to-Text system, built with FastAPI and integrating with the existing OpenAI Whisper transcription functionality.

## 🚀 Features

- **Single File Transcription**: Upload and transcribe individual audio files
- **Batch Transcription**: Process multiple audio files in a directory
- **Multilingual Support**: English, Hindi, Punjabi with auto-detection
- **Real-time Processing**: Fast transcription with progress tracking
- **Comprehensive API**: RESTful endpoints with OpenAPI documentation
- **Error Handling**: Robust error handling and validation
- **File Management**: Automatic temporary file cleanup

## 📋 API Endpoints

### Base Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information and available endpoints |
| `GET` | `/health` | Basic health check |
| `GET` | `/docs` | Interactive API documentation (Swagger UI) |
| `GET` | `/redoc` | Alternative API documentation |

### Transcription Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/transcription/transcribe` | Transcribe a single audio file |
| `POST` | `/transcription/transcribe-batch` | Batch transcribe multiple audio files |
| `GET` | `/transcription/supported-languages` | Get supported languages |
| `GET` | `/transcription/health` | Transcription service health check |

## 🛠️ Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Installation**:
   ```bash
   python -c "import fastapi, uvicorn; print('✅ Dependencies installed')"
   ```

## 🚀 Running the API

### Option 1: Direct Python Script
```bash
python run_api.py
```

### Option 2: Using the CLI
```bash
python main.py serve
```

### Option 3: Using Uvicorn Directly
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📖 API Usage Examples

### 1. Single File Transcription

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/transcription/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio_file.wav" \
  -F "language=auto" \
  -F "device=auto"
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/transcription/transcribe"
files = {'file': open('audio_file.wav', 'rb')}
data = {'language': 'auto', 'device': 'auto'}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Transcribed text: {result['full_text']}")
print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']}")
```

### 2. Batch Transcription

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/transcription/transcribe-batch" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "input_dir": "/path/to/audio/files",
    "output_dir": "/path/to/output",
    "language": "auto",
    "device": "auto"
  }'
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/transcription/transcribe-batch"
data = {
    "input_dir": "/path/to/audio/files",
    "output_dir": "/path/to/output",
    "language": "auto",
    "device": "auto"
}

response = requests.post(url, json=data)
result = response.json()

print(f"Total files: {result['total_files']}")
print(f"Successful: {result['successful_transcriptions']}")
print(f"Success rate: {result['success_rate']}%")
```

### 3. Get Supported Languages

```bash
curl -X GET "http://localhost:8000/transcription/supported-languages"
```

**Response**:
```json
{
  "languages": {
    "en": "English",
    "hi": "Hindi",
    "pa": "Punjabi"
  },
  "total_languages": 3
}
```

## 📁 Request/Response Models

### Transcription Request
```json
{
  "language": "auto",
  "device": "auto"
}
```

### Transcription Response
```json
{
  "success": true,
  "audio_file": "audio.wav",
  "language": "en",
  "full_text": "Hello, world!",
  "segments": [
    {
      "text": "Hello, world!",
      "start_time": 0.0,
      "end_time": 2.5,
      "confidence": 0.95
    }
  ],
  "confidence": 0.95,
  "processing_time": 1.23,
  "model": "openai/whisper-base",
  "device": "cpu",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Batch Transcription Request
```json
{
  "input_dir": "/path/to/audio",
  "output_dir": "/path/to/output",
  "language": "auto",
  "device": "auto"
}
```

### Batch Transcription Response
```json
{
  "success": true,
  "total_files": 10,
  "successful_transcriptions": 9,
  "failed_transcriptions": 1,
  "success_rate": 90.0,
  "output_dir": "/path/to/output",
  "processing_time": 45.67,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file based on `env.example`:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
MODEL_DEVICE=auto
DEFAULT_LANGUAGE=auto

# File Upload Limits
MAX_FILE_SIZE_MB=100
SUPPORTED_AUDIO_FORMATS=wav,mp3,m4a,flac,ogg,aac
```

### Supported Audio Formats
- **WAV** (.wav)
- **MP3** (.mp3)
- **M4A** (.m4a)
- **FLAC** (.flac)
- **OGG** (.ogg)
- **AAC** (.aac)

### File Size Limits
- **Maximum file size**: 100MB
- **Recommended**: < 50MB for optimal performance

## 🧪 Testing

### Run API Tests
```bash
# Start the API server in one terminal
python run_api.py

# Run tests in another terminal
python test_api.py
```

### Manual Testing
1. Start the API server
2. Open http://localhost:8000/docs
3. Use the interactive Swagger UI to test endpoints
4. Upload audio files and test transcription

## 📊 Monitoring

### Health Checks
- **Basic Health**: `/health`
- **Service Health**: `/transcription/health`
- **Model Status**: Check if Whisper model is loaded

### Logging
The API provides comprehensive logging:
- Request/response logging
- Error tracking
- Performance metrics
- Model initialization status

## 🚨 Error Handling

### Common HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid file, unsupported format)
- **413**: Payload Too Large (file too big)
- **500**: Internal Server Error (model issues, processing errors)

### Error Response Format
```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔒 Security Considerations

### CORS Configuration
- Currently allows all origins (`*`)
- Configure appropriately for production use
- Consider restricting to specific domains

### File Upload Security
- File type validation
- File size limits
- Temporary file cleanup
- Secure file handling

## 🚀 Production Deployment

### Using Gunicorn
```bash
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
```bash
# Disable debug mode
API_RELOAD=false
API_DEBUG=false

# Configure CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Security
API_SECRET_KEY=your-secret-key
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the `/docs` endpoint when running
- **Examples**: See `test_api.py` for usage examples

---

**Happy Transcribing! 🎤✨**

