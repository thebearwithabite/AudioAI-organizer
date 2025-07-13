#  AudioAI Organizer Examples

Comprehensive examples for different use cases and workflows.

##  Table of Contents

1. [Quick Start](#quick-start)
2. [Music Producer Workflow](#music-producer-workflow)
3. [Podcast Creator Setup](#podcast-creator-setup)
4. [Game Developer Organization](#game-developer-organization)
5. [AI Artist Custom Categories](#ai-artist-custom-categories)
6. [Advanced Batch Processing](#advanced-batch-processing)
7. [Learning System Integration](#learning-system-integration)

##  Quick Start

### Organize a Single File
```python
from audioai_organizer import AdaptiveAudioOrganizer

organizer = AdaptiveAudioOrganizer(
    openai_api_key="your-api-key",
    base_directory="/path/to/your/library"
)

# Process with preview
result = organizer.process_file_interactive(
    "mysterious_ambient_track.mp3", 
    dry_run=True
)
print(result)
```

### Organize Multiple Files
```python
audio_files = [
    "track1.mp3", "effect1.wav", "voice1.m4a"
]

# Smart mode - only asks when uncertain
organizer.set_interaction_mode('smart')
results = organizer.interactive_batch_process(
    audio_files, 
    confidence_threshold=0.7
)
```

## Music Producer Workflow
Perfect for organizing large sample libraries:
```python
import os
from pathlib import Path
from audioai_organizer import AdaptiveAudioOrganizer

def organize_sample_library():
    # Initialize for music production
    organizer = AdaptiveAudioOrganizer(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        base_directory="/Users/producer/SampleLibrary"
    )
    
    # Define music-specific categories
    music_categories = {
        "drums_acoustic": ["kick", "snare", "hihat", "cymbal", "tom"],
        "drums_electronic": ["808", "trap", "techno", "house", "dnb"],
        "bass_synth": ["reese", "wobble", "sub", "acid", "pluck"],
        "bass_acoustic": ["upright", "electric", "picked", "slapped"],
        "leads_warm": ["analog", "vintage", "moog", "prophet"],
        "leads_digital": ["fm", "wavetable", "granular", "modern"],
        "pads_ambient": ["strings", "choir", "atmospheric", "evolving"],
        "fx_transitions": ["sweep", "riser", "impact", "reverse"]
    }
    
    organizer.add_custom_categories(music_categories)
    
    # Set up for minimal interruption during creative flow
    organizer.set_interaction_mode('minimal')  # Only ask about very uncertain files
    
    # Scan common sample directories
    sample_dirs = [
        "/Users/producer/Downloads/Samples",
        "/Users/producer/Desktop/NewSamples",
        "/Applications/Ableton Live/Library/Samples"
    ]
    
    all_samples = []
    for directory in sample_dirs:
        if Path(directory).exists():
            all_samples.extend(
                organizer.find_audio_files_recursive(directory)
            )
    
    print(f" Found {len(all_samples)} samples to organize")
    
    # Process by BPM groups for better organization
    bpm_groups = organizer.group_files_by_tempo(all_samples)
    
    for bpm_range, files in bpm_groups.items():
        print(f" Processing {bpm_range} BPM samples ({len(files)} files)")
        organizer.interactive_batch_process(files, confidence_threshold=0.4)
    
    # Generate producer-friendly metadata
    df, spreadsheet = organizer.create_metadata_spreadsheet()
    
    # Add BPM-searchable tags
    df['BPM_Range'] = df['bpm'].apply(lambda x: f"{int(x//10)*10}-{int(x//10)*10+9}" if x else "Unknown")
    df['Key_Compatible'] = df['key'].apply(organizer.find_compatible_keys)
    
    df.to_excel(spreadsheet.replace('.xlsx', '_producer_enhanced.xlsx'), index=False)
    
    print(" Sample library organized for maximum creative flow!")

if __name__ == "__main__":
    organize_sample_library()
```

## Podcast Creator Setup
Organize voice recordings, SFX, and music for podcast production:
```python
def setup_podcast_library():
    organizer = AdaptiveAudioOrganizer(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        base_directory="/Users/podcaster/AudioAssets"
    )
    
    # Podcast-specific categories
    podcast_categories = {
        "voice_host": ["intro", "outro", "transition", "interview"],
        "voice_guest": ["interview", "soundbite", "quote"],
        "voice_narration": ["story", "explanation", "dramatic"],
        "music_intro": ["energetic", "welcoming", "branded"],
        "music_outro": ["memorable", "call_to_action", "fade"],
        "music_background": ["subtle", "neutral", "non_distracting"],
        "sfx_transition": ["swoosh", "chime", "musical", "pause"],
        "sfx_emphasis": ["alert", "notification", "highlight"],
        "sfx_ambient": ["office", "cafe", "nature", "urban"]
    }
    
    organizer.add_custom_categories(podcast_categories)
    
    # Always interactive for podcast precision
    organizer.set_interaction_mode('always')
    
    # Process recordings by episode
    episodes_dir = "/Users/podcaster/Recordings"
    for episode_folder in Path(episodes_dir).iterdir():
        if episode_folder.is_dir():
            print(f"️ Processing {episode_folder.name}")
            episode_files = organizer.find_audio_files_recursive(str(episode_folder))
            
            # Tag files with episode metadata
            for file_path in episode_files:
                result = organizer.process_file_interactive(
                    file_path,
                    additional_context=f"Episode: {episode_folder.name}"
                )
    
    # Create episode-searchable metadata
    df, _ = organizer.create_metadata_spreadsheet()
    df['Episode'] = df['original_path'].apply(lambda x: Path(x).parent.name)
    df['Duration_Minutes'] = df['duration_seconds'] / 60
    
    # Export podcast production sheet
    df.to_excel(
        organizer.base_dir / "04_METADATA_SYSTEM" / "podcast_production_database.xlsx",
        index=False
    )

if __name__ == "__main__":
    setup_podcast_library()
```

## Game Developer Organization
Structure audio assets for game development workflows:
```python
def organize_game_audio():
    organizer = AdaptiveAudioOrganizer(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        base_directory="/Users/gamedev/AudioAssets"
    )
    
    # Game development categories
    game_categories = {
        "ui_feedback": ["button", "hover", "confirm", "error", "notification"],
        "ui_ambient": ["menu_loop", "background_hum", "atmospheric"],
        "gameplay_action": ["jump", "shoot", "hit", "explosion", "pickup"],
        "gameplay_ambient": ["forest", "dungeon", "city", "space", "underwater"],
        "character_voice": ["player", "npc", "enemy", "narrator"],
        "character_foley": ["footsteps", "clothing", "breathing", "movement"],
        "music_adaptive": ["exploration", "combat", "victory", "defeat", "tension"],
        "music_linear": ["intro", "credits", "cutscene", "menu"]
    }
    
    organizer.add_custom_categories(game_categories)
    
    # Smart mode for game development efficiency
    organizer.set_interaction_mode('smart')
    
    # Process by game system
    systems = {
        "UI_Audio": "/Users/gamedev/RawAssets/UI",
        "Character_Audio": "/Users/gamedev/RawAssets/Characters", 
        "Environment_Audio": "/Users/gamedev/RawAssets/Environments",
        "Music": "/Users/gamedev/RawAssets/Music"
    }
    
    for system_name, system_path in systems.items():
        if Path(system_path).exists():
            print(f" Processing {system_name}")
            files = organizer.find_audio_files_recursive(system_path)
            
            results = organizer.interactive_batch_process(
                files,
                confidence_threshold=0.6,
                additional_context=f"Game System: {system_name}"
            )
    
    # Generate game-dev friendly metadata
    df, _ = organizer.create_metadata_spreadsheet()
    
    # Add game development specific columns
    df['File_Size_KB'] = df['file_path'].apply(
        lambda x: os.path.getsize(x) / 1024 if os.path.exists(x) else 0
    )
    df['Memory_Footprint'] = df['duration_seconds'] * df['sample_rate'] * 2 / 1024  # Rough estimate
    df['Loop_Candidate'] = df['classification'].apply(
        lambda x: 'ambient' in x or 'loop' in x
    )
    
    # Export for game engine import
    game_metadata = organizer.base_dir / "04_METADATA_SYSTEM" / "game_audio_database.xlsx"
    df.to_excel(game_metadata, index=False)
    
    print(" Game audio organized and ready for engine import!")

if __name__ == "__main__":
    organize_game_audio()