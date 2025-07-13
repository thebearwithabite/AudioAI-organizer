#!/bin/bash

# Quick install script for AudioAI Organizer dependencies
echo "🎵 Installing AudioAI Organizer dependencies..."

# Core audio processing libraries
pip install librosa
pip install mutagen
pip install openai
pip install pandas
pip install numpy

# Optional but recommended
pip install openpyxl  # For Excel export
pip install tqdm      # For progress bars
pip install jupyter   # If not already installed
pip install ipython   # For Jupyter audio playback

echo "✅ Installation complete!"
echo "🚀 You can now run: from audioai_organizer import AdaptiveAudioOrganizer"