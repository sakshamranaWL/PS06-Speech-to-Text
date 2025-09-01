# PS06 Speech-to-Text System

A multilingual speech recognition system using **OpenAI Whisper Large-v3** with automatic script conversion for high-quality transcription in English, Hindi, and Punjabi.

## 🚀 Key Features

- **Multilingual Support**: English, Hindi, and Punjabi transcription
- **Native Script Output**: Hindi in Devanagari, Punjabi in Gurmukhi
- **High Accuracy**: Uses OpenAI Whisper Large-v3 with script conversion
- **Preloaded Model**: Whisper Large-v3 model is preloaded for optimal performance
- **Language Hints**: Optional language hints ('hi', 'pa', 'en') for better accuracy
- **Task Configuration**: Uses task="transcribe" to keep output in original language
- **Simple API**: FastAPI application with comprehensive endpoints
- **Batch Processing**: Process multiple audio files efficiently
- **Real-time Processing**: Fast transcription with progress tracking

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd PS06-Speech-to-Text
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python3 transcribe.py --help
   ```

## 📖 Usage

### Command Line Interface

**Single file transcription**:
```bash
python3 transcribe.py data/audio/hindi1.mp3 --language hi
python3 transcribe.py data/audio/english1.mp3 --language en
python3 transcribe.py data/audio/punjabi1.mp3 --language pa
```

**Available options**:
- `--language`: Language code (`en`, `hi`, `pa`, `auto`)
- `--device`: Device to use (`auto`, `cpu`, `cuda`)

### API Usage

**Start the API server**:
```bash
python3 start_api.py
```

**API endpoints**:
- `GET /`: API information
- `POST /transcribe`: Transcribe single audio file
- `POST /transcribe-batch`: Batch transcribe multiple files
- `GET /supported-languages`: Get language information
- `GET /health`: Health check
- `GET /docs`: Interactive API documentation

**Example API usage**:
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@audio_file.wav" \
  -F "language=hi"
```

## 🔧 How It Works

### Language-Specific Processing

1. **English**: Uses OpenAI Whisper directly
2. **Hindi**: Uses Whisper + Hindi Devanagari script converter
3. **Punjabi**: Uses Whisper + Punjabi Gurmukhi script converter

### Script Conversion

The system automatically detects when Whisper outputs Urdu/Arabic script and converts it to the appropriate native script:
- **Hindi audio** → **Devanagari script** (नमस्ते)
- **Punjabi audio** → **Gurmukhi script** (ਸਤ ਸ੍ਰੀ ਅਕਾਲ)

## 📊 Technical Details

### Models Used

| Language | Model | Script Output |
|----------|-------|---------------|
| **English** | OpenAI Whisper Large-v3 | English |
| **Hindi** | OpenAI Whisper Large-v3 + Converter | Devanagari |
| **Punjabi** | OpenAI Whisper Large-v3 + Converter | Gurmukhi |

### Performance

- **Processing Speed**: ~8-11 seconds per audio file
- **Accuracy**: High confidence scores (0.7-0.8)
- **Memory Usage**: Efficient with automatic cleanup
- **File Support**: WAV, MP3, M4A, FLAC, OGG, AAC

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Test with sample audio**:
   ```bash
   python3 transcribe.py data/audio/hindi1.mp3 --language hi
   ```

3. **Start API server**:
   ```bash
   python3 start_api.py
   ```

4. **Access API documentation**:
   - Open: http://localhost:8000/docs
   - Interactive Swagger UI

5. **Test the API**:
   ```bash
   python3 test_api.py
   ```

## 📁 Project Structure

```
PS06-Speech-to-Text/
├── transcribe.py          # Main transcription functionality
├── api.py                 # FastAPI application
├── start_api.py           # API startup script
├── test_api.py            # API testing script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── LICENSE               # MIT License
└── data/
    └── audio/            # Sample audio files
```

## 🔍 Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure sufficient disk space and internet connection
2. **Memory issues**: Use CPU device for lower memory usage
3. **Audio format issues**: Convert to supported formats (WAV, MP3, etc.)

### Performance Tips

- Use specific language codes (`hi`, `pa`) instead of `auto` for better accuracy
- Use GPU (`cuda`) if available for faster processing
- Process audio files in batch for efficiency

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
