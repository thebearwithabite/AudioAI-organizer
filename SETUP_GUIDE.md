#  AudioAI Organizer Setup Guide

Complete step-by-step setup instructions for AudioAI Organizer.

##  Prerequisites

### System Requirements
- **Python 3.8 or higher** (3.9+ recommended)
- **4GB+ RAM** (8GB+ for large libraries)
- **OpenAI API key** with GPT-4 access
- **Audio files** in common formats (MP3, WAV, AIFF, M4A, FLAC, OGG)

### Supported Platforms
- ✅ **macOS** (Intel & Apple Silicon)
- ✅ **Windows** 10/11
- ✅ **Linux** (Ubuntu, Debian, CentOS)

## ️ Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/thebearwithabite/AudioAI-organizer.git
cd audioai-organizer
```

### 2. Set Up Python Environment (Recommended)
```bash
# Create virtual environment
python -m venv audioai_env

# Activate it
# On macOS/Linux:
source audioai_env/bin/activate
# On Windows:
audioai_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**If you encounter issues with librosa/audio dependencies:**

**macOS:**
```bash
brew install portaudio
pip install librosa
```

**Ubuntu/Debian:**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip install librosa
```

**Windows:**
```bash
# Usually installs without issues, but if needed:
pip install --upgrade setuptools wheel
pip install librosa
```

### 4. Get Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new secret key
3. Copy the key (starts with `sk-`)

**Set your API key:**

**Option A: Environment Variable (Recommended)**
```bash
# macOS/Linux - add to ~/.bashrc or ~/.zshrc:
export OPENAI_API_KEY="sk-your-actual-api-key-here"

# Windows - add to system environment variables or:
set OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Option B: Direct in Code**
```python
organizer = AdaptiveAudioOrganizer(
    openai_api_key="sk-your-actual-api-key-here",
    base_directory="/path/to/your/audio/library"
)
```

### 5. Set Up Your Audio Library Directory

Choose a directory for your organized audio library:
```bash
# Example locations:
# macOS: /Users/yourusername/AudioLibrary
# Windows: C:\Users\yourusername\AudioLibrary  
# Linux: /home/yourusername/AudioLibrary
```

The system will automatically create this structure:
```
YOUR_AUDIO_LIBRARY/
├── 01_UNIVERSAL_ASSETS/
├── THEMATIC_COLLECTIONS/
├── 04_METADATA_SYSTEM/
└── TO_SORT/
```

##  Test Your Installation

### Quick Test Script
Create a file called `test_audioai.py`:

```python
#!/usr/bin/env python3
from audioai_organizer import AdaptiveAudioOrganizer
import os

# Test configuration
API_KEY = os.getenv('OPENAI_API_KEY') or "your-api-key-here"
LIBRARY_PATH = "/path/to/test/library"  # Change this path

def test_audioai():
    print(" Testing AudioAI Organizer...")
    
    try:
        # Initialize organizer
        organizer = AdaptiveAudioOrganizer(
            openai_api_key=API_KEY,
            base_directory=LIBRARY_PATH
        )
        print("✅ Initialization successful!")
        
        # Test audio extensions
        print(f" Supported formats: {organizer.audio_extensions}")
        
        # Test folder mapping
        print(f"️ Base categories: {len(organizer.base_categories)} categories")
        
        print(" AudioAI is ready to use!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_audioai()
```

Run the test:
```bash
python test_audioai.py
```

### Test with a Sample Audio File

```python
# Test with a single audio file
from audioai_organizer import AdaptiveAudioOrganizer

organizer = AdaptiveAudioOrganizer(
    openai_api_key="your-api-key",
    base_directory="/path/to/your/library"
)

# Test with dry run (won't actually move files)
result = organizer.process_file_interactive(
    "/path/to/test/audio.mp3", 
    dry_run=True
)

print("Test result:", result)
```

##  Configuration Options

### Setting Interaction Modes

```python
# Smart mode (recommended) - asks when uncertain
organizer.set_interaction_mode('smart')    # 70% confidence threshold

# Minimal interruption mode
organizer.set_interaction_mode('minimal')  # 40% confidence threshold

# Always ask for perfect accuracy
organizer.set_interaction_mode('always')   # 100% threshold

# Fully automatic for bulk processing
organizer.set_interaction_mode('never')    # 0% threshold
```

### Custom Category Configuration

```python
# Add your specific categories
custom_categories = {
    "music_genre_house": ["deep", "tech", "progressive", "minimal"],
    "sfx_game_audio": ["ui_sounds", "ambient", "action", "menu"],
    "voice_podcast": ["intro", "outro", "transition", "interview"]
}

organizer.add_custom_categories(custom_categories)
```

##  Troubleshooting

### Common Issues

**1. "No module named 'librosa'"**
```bash
# Try installing with conda instead:
conda install librosa -c conda-forge

# Or force reinstall:
pip uninstall librosa
pip install librosa --no-cache-dir
```

**2. "OpenAI API key not found"**
```bash
# Check your environment variable:
echo $OPENAI_API_KEY

# Or set it temporarily:
export OPENAI_API_KEY="sk-your-key"
```

**3. "Permission denied" errors**
```bash
# Make sure Python has write access to your library directory:
chmod 755 /path/to/your/audio/library

# On Windows, run Command Prompt as Administrator
```

**4. "Audio file not supported"**
- Check file format: `file your_audio.mp3`
- Ensure file isn't corrupted
- Try with a different audio file first

**5. Memory issues with large files**
```python
# For very large audio files, process in smaller batches:
organizer.interactive_batch_process(
    audio_files[:10],  # Process 10 at a time
    confidence_threshold=0.7
)
```

### Getting Help

**Check system info:**
```python
import sys, librosa, pandas, openai
print(f"Python: {sys.version}")
print(f"Librosa: {librosa.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"OpenAI: {openai.__version__}")
```

**Enable debug mode:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your organizer commands
```

**Contact Support:**
-  Email: rt@papersthatdream.com
-  [GitHub Issues](https://github.com/thebearwithabite/AudioAI-organizer/issues)
-  [Full Documentation](https://docs.papersthatdreamn.com)

##  Next Steps

Once setup is complete:

1. **Start small**: Test with 5-10 audio files first
2. **Use dry_run=True**: Preview what will happen before moving files
3. **Check the metadata**: Review generated spreadsheets
4. **Customize categories**: Add your specific organization needs
5. **Enable learning**: Let the system adapt to your preferences

**Happy organizing!** ✨