#!/usr/bin/env python3
"""
Basic AudioAI Organizer Usage Example

This example shows the most common use case: organizing a small collection
of audio files with interactive classification.
"""

from audioai_organizer import AdaptiveAudioOrganizer
import os
from pathlib import Path

def main():
    # Configuration
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        print("❌ Please set your OPENAI_API_KEY environment variable")
        return
    
    # Set up your paths
    LIBRARY_PATH = "/path/to/your/audio/library"  # Change this!
    TEST_FILES = [
        "/path/to/test/file1.mp3",
        "/path/to/test/file2.wav",
        # Add your test files here
    ]
    
    print(" AudioAI Organizer - Basic Usage Example")
    print("=" * 50)
    
    # Initialize organizer
    print("Initializing AudioAI...")
    organizer = AdaptiveAudioOrganizer(
        openai_api_key=API_KEY,
        base_directory=LIBRARY_PATH
    )
    
    # Set smart interaction mode (asks when uncertain)
    organizer.set_interaction_mode('smart')
    
    # Process files one by one
    for audio_file in TEST_FILES:
        if not Path(audio_file).exists():
            print(f"⚠️  File not found: {audio_file}")
            continue
            
        print(f"\n Processing: {Path(audio_file).name}")
        
        # Start with dry run to see what would happen
        result = organizer.process_file_interactive(
            audio_file, 
            dry_run=True
        )
        
        print(f" Analysis: {result}")
        
        # Uncomment to actually move files:
        # result = organizer.process_file_interactive(audio_file, dry_run=False)
    
    # Generate metadata spreadsheet
    print("\n Generating metadata spreadsheet...")
    df, spreadsheet_path = organizer.create_metadata_spreadsheet()
    print(f" Metadata saved to: {spreadsheet_path}")
    
    print("\n✅ Basic processing complete!")

if __name__ == "__main__":
    main()