#  AudioAI Organizer

**Intelligent audio library organization with AI-powered analysis, interactive classification, and adaptive learning.**

Transform your chaotic audio collections into intelligently organized, searchable libraries with AI that actually *listens* to your content and learns your creative patterns.

![GitHub stars](https://img.shields.io/github/stars/thebearwithabite/audioai-organizer)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)

## ✨ What Makes This Special

**AudioAI doesn't just sort files - it understands them.**

-  **Listens to actual audio content** using advanced spectral analysis
-  **Learns your organization patterns** and discovers new categories organically  
-  **Interactive classification** asks for help when uncertain, improving over time
-  **Semantic filename preservation** keeps meaning while adding rich metadata
-  **Confidence-based processing** - auto-handles obvious files, asks about edge cases
-  **Adaptive learning system** gets smarter with every classification

##  Quick Demo

### Before AudioAI:
```
downloads/
├── 88bpm_play playful_childlike_beat_ES_February_Moon.mp3
├── pulsing_signals_digital_space_Rhythmania.mp3  
├── out-of-breath-male-176715.mp3
└── UK-Asian_Young_Female_Voice_35.wav
```

### After AudioAI:
```
01_UNIVERSAL_ASSETS/
├── MUSIC_LIBRARY/by_mood/contemplative/
│   └── playful_childlike_February_Moon_Instrumental_MUS_88bpm_CONT_E7.mp3
├── SFX_LIBRARY/by_category/technology/
│   └── pulsing_signals_digital_space_Rhythmania_SFX_TECH_90bpm_MYST_E6.mp3
└── SFX_LIBRARY/by_category/human_elements/
    ├── Male_Out-of-Breath_SFX_176715.mp3
    └── UK-Asian_Young_Female_Voice_35.wav

04_METADATA_SYSTEM/
└── audio_metadata_20250708.xlsx  # Comprehensive searchable database
```

**Plus:** Confidence scores, AI reasoning, cross-references, and learning statistics!

## 📺 Demo Video

[![Watch the AudioAI Organizer Demo](https://img.youtube.com/vi/DKdwy9dbJ3g/0.jpg)](https://www.youtube.com/watch?v=DKdwy9dbJ3g)

## ️ Installation

### Prerequisites
```bash
# Python 3.8+ required
pip install openai librosa mutagen pandas openpyxl numpy pathlib
```

### Quick Setup
```bash
git clone https://github.com/yourusername/audioai-organizer.git
cd audioai-organizer
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"
```

### First Run
```python
from audioai_organizer import AdaptiveAudioOrganizer

# Initialize with your library path
organizer = AdaptiveAudioOrganizer(
    openai_api_key="your-api-key-here",
    base_directory="/path/to/your/audio/library"
)

# Process a single file interactively
result = organizer.process_file_interactive("test_audio.mp3", dry_run=True)

# Batch process with smart interaction (recommended)
audio_files = ["/path/to/audio1.mp3", "/path/to/audio2.wav"]
organizer.interactive_batch_process(audio_files, confidence_threshold=0.7)
```

##  Core Features

###  **Adaptive AI Classification**
- **Audio content analysis**: BPM, brightness, texture, energy levels
- **Pattern recognition**: Learns your specific organization preferences
- **Interactive learning**: Asks targeted questions to improve accuracy
- **Confidence scoring**: Auto-processes obvious files, flags uncertain ones

###  **Intelligent Audio Understanding** 
- **Tempo detection**: Precise BPM extraction for rhythm-based organization
- **Mood analysis**: Emotional classification (contemplative, mysterious, energetic)
- **Content type detection**: Music vs SFX vs voice with high accuracy
- **Spectral analysis**: Brightness, texture, and tonal characteristics

###  **Dynamic Organization System**
- **Semantic folder structures**: Organized by mood, energy, and purpose
- **Cross-reference system**: Files can belong to multiple relevant categories
- **Automatic folder creation**: Discovers new categories from your content
- **Filename enhancement**: Preserves meaning while adding rich metadata

###  **Comprehensive Tracking**
- **Searchable metadata**: Excel spreadsheets with full analysis data
- **Learning statistics**: Track system improvement over time
- **Original filename preservation**: Complete traceability
- **Confidence and reasoning**: Understand every AI decision

##  Perfect For

###  **Music Producers**
*"I had 10,000 samples scattered everywhere. AudioAI organized them by BPM, mood, and energy in 2 hours. Now I find the perfect 128bpm dark ambient pad instantly."*

###  **Content Creators** 
*"Managing voice samples, SFX, and music was chaos. The semantic filename preservation means I never lose track of what files actually contain."*

###  **Game Developers**
*"The interactive classification caught edge cases I missed. The AI learned our specific sound design categories and now auto-sorts 95% of new assets."*

###  **AI Artists & Researchers**
*"Building audio libraries for AI consciousness storytelling. The system understands emotional context and organizes themes like 'digital consciousness' and 'memory formation'."*

##  Advanced Usage

### **Smart Interaction Modes**
```python
# Smart mode - asks when uncertain (recommended)
organizer.set_interaction_mode('smart')    # 70% confidence threshold

# Minimal mode - only very uncertain files  
organizer.set_interaction_mode('minimal')  # 40% confidence threshold

# Always interactive - maximum accuracy
organizer.set_interaction_mode('always')   # 100% threshold

# Fully automatic - bulk processing
organizer.set_interaction_mode('never')    # 0% threshold
```

### **Learning System Integration**
```python
# View learning statistics
organizer.show_learning_stats()

# Export classifications for backup
learning_data = organizer.export_learning_data()

# Import existing classifications
organizer.import_classifications("previous_library.json")

# Force learning update after manual corrections
organizer.update_learning_patterns()
```

### **Custom Category Development**
```python
# Define custom organization patterns
custom_categories = {
    "music_electronic": ["energetic", "euphoric", "dark", "minimal"],
    "sfx_nature": ["calming", "organic", "flowing", "textural"],
    "voice_ai": ["synthetic", "robotic", "processed", "emotional"]
}

organizer.add_custom_categories(custom_categories)
```

### **Batch Processing with Audio Playback**
```python
# Process with real-time audio preview (Jupyter/IPython)
organizer.interactive_batch_process(
    file_list,
    confidence_threshold=0.7,
    play_audio=True  # Hear uncertain files before classifying
)
```

## 🚀 Advanced Interactive Features (v16+)

### Human-in-the-Loop with Audio Preview
- When the AI is less than 70% confident, it will prompt you for confirmation and play a 30-second audio preview (in Jupyter/IPython).
- You can accept, modify, or skip the classification, ensuring maximum accuracy and control.

### Emergency Logging for Recovery
- Every file move and filename change is logged in the metadata spreadsheet, including original and new filenames/paths.
- This allows you to recover or trace any file, even after large batch operations.

### Environment Variable Setup
- For security and portability, set your API key and base directory using environment variables:
  ```bash
  export OPENAI_API_KEY="sk-your-key-here"
  export AUDIOAI_BASE_DIRECTORY="/path/to/your/audio/library"
  ```

### Single-File Test Mode
- Test the system on a single file (with audio preview and user feedback) before running batch processing:
  ```python
  from audioai_organizer import AdaptiveAudioOrganizer
  import os
  
  organizer = AdaptiveAudioOrganizer(
      openai_api_key=os.getenv('OPENAI_API_KEY'),
      base_directory=os.getenv('AUDIOAI_BASE_DIRECTORY')
  )
  
  result = organizer.process_file_interactive("/path/to/test/file.mp3", dry_run=True)
  ```

### Batch Processing with Audio Preview
- Process multiple files, with human-in-the-loop and audio preview for uncertain files:
  ```python
  audio_files = ["/path/to/audio1.mp3", "/path/to/audio2.wav"]
  results = organizer.interactive_batch_process(audio_files, confidence_threshold=0.7, dry_run=True)
  ```
- The system will prompt you and play audio for files where the AI is unsure.

### Emergency Recovery
- The metadata spreadsheet (`audio_metadata_YYYYMMDD.xlsx`) contains all original and new filenames/paths for every processed file.
- Use this log to recover or trace any file if needed.

##  Library Structure

AudioAI creates an intelligent, expandable folder structure:

```
YOUR_AUDIO_LIBRARY/
├── 01_UNIVERSAL_ASSETS/
│   ├── MUSIC_LIBRARY/
│   │   └── by_mood/
│   │       ├── contemplative/
│   │       ├── tension_building/
│   │       ├── mysterious/
│   │       └── wonder_discovery/
│   ├── SFX_LIBRARY/
│   │   └── by_category/
│   │       ├── consciousness/
│   │       ├── human_elements/
│   │       ├── environmental/
│   │       └── technology/
│   └── VOICE_ELEMENTS/
│       ├── narrator_banks/
│       ├── processed_vocals/
│       └── character_voices/
├── THEMATIC_COLLECTIONS/
│   ├── human_machine_dialogue/
│   ├── digital_consciousness/
│   └── emergence_awakening/
├── 04_METADATA_SYSTEM/
│   ├── learning_data.pkl
│   ├── discovered_categories.json
│   └── audio_metadata_YYYYMMDD.xlsx
└── TO_SORT/
    └── [unprocessed files]
```

##  Technical Deep Dive

### **Audio Analysis Pipeline**
1. **Spectral Analysis**: Librosa extracts tempo, brightness, texture
2. **Content Classification**: ML distinguishes music/SFX/voice
3. **Mood Detection**: Energy and harmonic analysis for emotional context
4. **Pattern Recognition**: Compares against learned user preferences
5. **Confidence Scoring**: Determines if human input needed

### **Learning Algorithm**
- **User feedback integration**: Every correction improves future classifications
- **Pattern discovery**: Identifies recurring organization themes
- **Category evolution**: Automatically discovers new meaningful groupings
- **Confidence calibration**: Learns when to ask vs auto-process

### **Filename Enhancement System**
- **Semantic preservation**: Keeps original meaning and context
- **Metadata integration**: Adds BPM, energy, mood, classification codes
- **Collision handling**: Smart numbering for duplicate semantic content
- **Reversibility**: Complete traceability back to original names

##  Contributing

We'd love your help making AudioAI even better!

### **Most Wanted Features**
- [ ] DAW integration (Ableton Live, Logic Pro, etc.)
- [ ] Cloud storage sync (Google Drive, Dropbox)
- [ ] Multi-language filename support
- [ ] Advanced genre-specific classification models
- [ ] Web interface for remote library management
- [ ] Audio similarity clustering
- [ ] Collaborative library sharing

### **Bug Reports**
Found an issue? Please include:
- Audio file format and duration
- Full error traceback
- Your system info (OS, Python version)
- Example file (if shareable)

### **Development Setup**
```bash
git clone https://github.com/thebearwithabite/AudioAI-Organizer.git
cd audioai-organizer
pip install -e ".[dev]"
pytest tests/
```

##  Performance & Scaling

- **Processing speed**: ~50-100 files per hour (depending on interaction mode)
- **Library size**: Tested with 50,000+ file libraries
- **Memory usage**: ~200MB for typical audio analysis
- **Storage overhead**: ~5MB metadata per 1000 files

##  Privacy & Security

- **Local processing**: All audio analysis happens on your machine
- **API usage**: Only sends text descriptions to OpenAI, never audio files
- **No data collection**: Your library organization stays completely private
- **Offline mode**: Core audio analysis works without internet

##  License

MIT License - see [LICENSE](LICENSE) for details.

Built with ❤️ for the audio community. 🐻

##  Acknowledgments

- **librosa** - Incredible audio analysis capabilities
- **OpenAI** - GPT-4 language understanding  
- **pandas** - Data management and export
- **mutagen** - Audio metadata extraction
- **The audio community** - For inspiring better organization tools

---

##  Star this repo if AudioAI transformed your audio workflow!

**Questions? Ideas? Success stories?**
-  [Open an issue](https://github.com/thebearwithabite/AudioAI-Organizer/issues)
-  Email: [rt@papersthatdream.com](rt@papersthatdream.com)
-  Substack:  [rt-max.substack.com](https://rt-max.substack.com)

*From audio chaos to intelligent organization. AudioAI learns, adapts, and grows with your creative vision.*