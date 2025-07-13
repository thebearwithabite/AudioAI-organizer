# This is a custom categories example.
from audioai_organizer import AdaptiveAudioOrganizer

categories = {
    'music': ['mp3', 'wav'],
    'sfx': ['wav', 'aiff'],
}

organizer = AdaptiveAudioOrganizer(categories=categories)
organizer.organize_audio('path/to/your/audio/files')