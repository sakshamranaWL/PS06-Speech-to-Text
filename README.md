# 🎤 PS06 Speech-to-Text System

**Multilingual Speech Recognition using AI4Bharat IndicConformer Model from AI Kosh**

## 🌟 Features

- **Multilingual ASR**: English, Hindi, and Punjabi support
- **AI4Bharat IndicConformer**: State-of-the-art model for Indian languages
- **Audio Processing**: Support for WAV, MP3, M4A, FLAC formats
- **Batch Processing**: Efficient processing of multiple audio files
- **Evaluation Metrics**: WER/CER calculation and analysis
- **Simple CLI**: Easy-to-use command-line interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model
```bash
# Download AI4Bharat IndicConformer model from AI Kosh
make download-model
```

### 3. Transcribe Audio
```bash
# Single file
python transcribe.py --audio path/to/audio.wav --language hi

# Batch processing
python transcribe.py --input-dir data/audio --output-dir data/transcripts
```

## 📁 Project Structure

```
PS06-Speech-to-Text/
├── transcribe.py          # Main transcription script
├── download_models.py     # Model download utility
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Audio files and transcripts
│   ├── audio/           # Input audio files
│   └── transcripts/     # Output transcripts
└── models/              # Downloaded models
```

## 🎯 Supported Languages

- **English (en)**: General English speech
- **Hindi (hi)**: Hindi speech with Devanagari script
- **Punjabi (pa)**: Punjabi speech with Gurmukhi script

## 🔧 Configuration

The system uses the AI4Bharat IndicConformer model by default:

```yaml
Model: ai4bharat/indic-conformer-v1
Languages: [hi, en, pa]
Source: AI Kosh / HuggingFace
```

## 📊 Output Format

Transcripts are saved in JSON format:

```json
{
  "audio_file": "audio.wav",
  "language": "hi",
  "segments": [
    {
      "text": "मेरा नाम राहुल है",
      "start_time": 0.0,
      "end_time": 2.5,
      "confidence": 0.95
    }
  ],
  "metadata": {
    "model": "ai4bharat/indic-conformer-v1",
    "processing_time": 15.2
  }
}
```

## 🧪 Testing

Test the system with sample audio files:

```bash
# Test Hindi transcription
python transcribe.py --audio test_audio/hindi_sample.wav --language hi

# Test English transcription  
python transcribe.py --audio test_audio/english_sample.wav --language en

# Test Punjabi transcription
python transcribe.py --audio test_audio/punjabi_sample.wav --language pa
```

## 📈 Performance

- **Accuracy**: State-of-the-art performance on Indic languages
- **Speed**: Real-time processing capability
- **Memory**: Optimized for efficient resource usage
- **Scalability**: Batch processing for large datasets

## 🛠️ Available Commands

```bash
# System setup
make install              # Install dependencies
make test                 # Run system tests
make download-model       # Download AI4Bharat model
make verify-model         # Verify downloaded model

# Transcription
python transcribe.py --help                    # Show help
python transcribe.py --audio file.wav --language hi  # Single file
python transcribe.py --input-dir audio/ --output-dir transcripts/  # Batch
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **AI4Bharat** for the IndicConformer model
- **AI Kosh** for model distribution
- **Open Source Community** for supporting libraries
