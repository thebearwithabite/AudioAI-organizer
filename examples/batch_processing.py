#!/usr/bin/env python3
"""
Batch Processing Example for AudioAI Organizer

Shows how to efficiently process large collections of audio files
with different interaction modes and confidence thresholds.
"""

from audioai_organizer import AdaptiveAudioOrganizer
import os
from pathlib import Path
import glob

def find_audio_files(directories):
    """Find all audio files in given directories."""
    audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
    audio_files = []
    
    for directory in directories:
        for ext in audio_extensions:
            pattern = f"{directory}/**/*{ext}"
            files = glob.glob(pattern, recursive=True)
            audio_files.extend(files)
    
    return sorted(set(audio_files))  # Remove duplicates and sort

def main():
    # Configuration
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    LIBRARY_PATH = os.getenv("AUDIOAI_BASE_DIRECTORY", "")
    if not LIBRARY_PATH:
        print("❌ Please set AUDIOAI_BASE_DIRECTORY environment variable")
        print("   export AUDIOAI_BASE_DIRECTORY='/path/to/your/audio/library'")
        return
    
    # Directories to scan for audio files
    raw_scan = os.getenv("AUDIOAI_SCAN_DIRS", "")
    SCAN_DIRECTORIES = [p.strip() for p in raw_scan.split(",") if p.strip()] or [LIBRARY_PATH]
    
    print(" AudioAI Organizer - Batch Processing Example")
    print("=" * 55)
    
    # Find all audio files
    print(" Scanning for audio files...")
    audio_files = find_audio_files(SCAN_DIRECTORIES)
    print(f" Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("⚠️  No audio files found. Check your scan directories.")
        return
    
    # Initialize organizer
    print("\n Initializing AudioAI...")
    organizer = AdaptiveAudioOrganizer(
        openai_api_key=API_KEY,
        base_directory=LIBRARY_PATH
    )
    
    # Show current learning stats
    organizer.show_learning_stats()
    
    # Processing options
    print("\n⚙️  Processing Options:")
    print("1. Smart mode (asks when uncertain - recommended)")
    print("2. Minimal mode (only very uncertain files)")
    print("3. Automatic mode (no questions, fastest)")
    print("4. Always ask (maximum accuracy)")
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    mode_map = {
        '1': ('smart', 0.7),
        '2': ('minimal', 0.4), 
        '3': ('never', 0.0),
        '4': ('always', 1.0)
    }
    
    if choice not in mode_map:
        print("Invalid choice, using smart mode")
        choice = '1'
    
    mode_name, threshold = mode_map[choice]
    organizer.set_interaction_mode(mode_name)
    
    print(f"\n Using {mode_name} mode (confidence threshold: {threshold})")
    
    # Ask about dry run
    dry_run = input("\nStart with dry run? (y/n): ").strip().lower() == 'y'
    
    if dry_run:
        print("\n Running in DRY RUN mode - no files will be moved")
    else:
        print("\n LIVE MODE - files will be organized")
    
    # Process files in batches
    batch_size = 20
    total_files = len(audio_files)
    
    for i in range(0, total_files, batch_size):
        batch = audio_files[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_files + batch_size - 1) // batch_size
        
        print(f"\n Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        try:
            results = organizer.interactive_batch_process(
                batch,
                confidence_threshold=threshold,
                dry_run=dry_run
            )
            
            # Show batch results
            successful = len([r for r in results if r.get('success', False)])
            print(f"✅ Batch {batch_num} complete: {successful}/{len(batch)} successful")
            
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error in batch {batch_num}: {e}")
            continue
    
    # Generate final metadata
    print("\n Generating comprehensive metadata...")
    df, spreadsheet_path = organizer.create_metadata_spreadsheet()
    print(f" Metadata saved to: {spreadsheet_path}")
    
    # Show final learning stats
    print("\n Final Learning Statistics:")
    organizer.show_learning_stats()
    
    print("\n Batch processing complete!")
    print(f" Your organized library: {LIBRARY_PATH}")

if __name__ == "__main__":
    main()