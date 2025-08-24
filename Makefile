# PS06 Speech-to-Text System Makefile

.PHONY: help install test clean download-model verify-model

help:  ## Show this help message
	@echo "PS06 Speech-to-Text System - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

test:  ## Run system tests
	python test_system.py

clean:  ## Clean temporary files
	rm -rf data/transcripts/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download-model:  ## Download AI4Bharat IndicConformer model
	python download_models.py --download

verify-model:  ## Verify downloaded model
	python download_models.py --verify

model-info:  ## Show model information
	python download_models.py --info

transcribe-help:  ## Show transcription help
	python transcribe.py --help

demo:  ## Show system demo
	python demo.py

# Quick start
quick-start: install test  ## Quick start setup
	@echo "Quick start completed!"
	@echo "Next steps:"
	@echo "1. Download model: make download-model"
	@echo "2. Verify model: make verify-model"
	@echo "3. Test transcription: python transcribe.py --help"
