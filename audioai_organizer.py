# Cell: Import required libraries
import os
import shutil
import json
import time
from pathlib import Path
from openai import OpenAI
import mutagen
from collections import defaultdict
import pickle
from datetime import datetime

class AdaptiveAudioOrganizer:
    def __init__(self, openai_api_key, base_directory):
        self.client = OpenAI(api_key=openai_api_key)
        self.base_dir = Path(base_directory)
        
        # Learning system files
        self.learning_data_file = self.base_dir / "04_METADATA_SYSTEM" / "learning_data.pkl"
        self.discovered_categories_file = self.base_dir / "04_METADATA_SYSTEM" / "discovered_categories.json"
        
        # Load existing learning data
        self.learning_data = self.load_learning_data()
        self.discovered_categories = self.load_discovered_categories()
        
        # Base categories that can expand
        self.base_categories = {
            "music_ambient": ["contemplative", "tension_building", "wonder_discovery", "melancholic", "mysterious"],
            "sfx_consciousness": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_human": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_environmental": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_technology": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "voice_element": ["contemplative", "melancholic", "mysterious", "dramatic_punctuation"],
        }
        
        # Dynamic folder mapping that grows over time
        self.folder_map = self.build_dynamic_folder_map()
        self.audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
        
    def load_learning_data(self):
        """Load historical classification data"""
        if self.learning_data_file.exists():
            with open(self.learning_data_file, 'rb') as f:
                return pickle.load(f)
        return {
            'classifications': [],
            'user_corrections': [],
            'patterns': defaultdict(list),
            'filename_patterns': defaultdict(list)
        }
    
    def save_learning_data(self):
        """Save learning data for future use"""
        self.learning_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.learning_data_file, 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def load_discovered_categories(self):
        """Load dynamically discovered categories"""
        if self.discovered_categories_file.exists():
            with open(self.discovered_categories_file, 'r') as f:
                return json.load(f)
        return {
            'new_moods': [],
            'new_categories': [],
            'new_themes': [],
            'frequency_counts': defaultdict(int)
        }
    
    def save_discovered_categories(self):
        """Save discovered categories"""
        self.discovered_categories_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.discovered_categories_file, 'w') as f:
            json.dump(self.discovered_categories, f, indent=2)
    
    def build_dynamic_folder_map(self):
        """Build folder mapping that includes discovered categories"""
        base_map = {
            "music_ambient + contemplative": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/contemplative/",
            "music_ambient + tension_building": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/tension_building/",
            "music_ambient + wonder_discovery": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/wonder_discovery/",
            "music_ambient + melancholic": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/melancholic/",
            "music_ambient + mysterious": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/mysterious/",
            "sfx_consciousness + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/thought_processing/",
            "sfx_consciousness + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/awakening_emergence/",
            "sfx_consciousness + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/memory_formation/",
            "sfx_human + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/breathing_heartbeat/",
            "sfx_human + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/emotional_responses/",
            "sfx_human + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/environmental_human/",
            "sfx_environmental + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/time_space/",
            "sfx_environmental + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/transformation/",
            "sfx_environmental + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/connection_bridging/",
            "voice_element + contemplative": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/narrator_banks/",
            "voice_element + melancholic": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/processed_vocals/",
            "voice_element + mysterious": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/vocal_textures/",
            "voice_element + dramatic_punctuation": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/character_voices/",
            "default": "01_UNIVERSAL_ASSETS/UNSORTED/"
        }
        
        # Add discovered categories
        for category in self.discovered_categories.get('new_categories', []):
            for mood in self.discovered_categories.get('new_moods', []):
                key = f"{category} + {mood}"
                if key not in base_map:
                    # Create new folder path based on category type
                    if category.startswith('music_'):
                        base_map[key] = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
                    elif category.startswith('sfx_'):
                        sfx_type = category.replace('sfx_', '')
                        base_map[key] = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
                    elif category.startswith('voice_'):
                        base_map[key] = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
                    else:
                        base_map[key] = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        return base_map
    
    def get_audio_metadata(self, file_path):
        """Extract audio metadata using mutagen"""
        try:
            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                duration = audio_file.info.length
                return {
                    'duration': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'bitrate': getattr(audio_file.info, 'bitrate', 'Unknown'),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 'Unknown')
                }
        except Exception as e:
            print(f"Could not read metadata for {file_path}: {e}")
        return {'duration': 'Unknown', 'bitrate': 'Unknown', 'sample_rate': 'Unknown'}
    
    def analyze_filename_patterns(self, filename):
        """Extract patterns from filename that might indicate content type"""
        patterns = []
        filename_lower = filename.lower()
        
        # Common audio descriptors
        descriptors = {
            'ambient': ['ambient', 'atmosphere', 'atmos', 'background', 'bg'],
            'percussion': ['drum', 'beat', 'perc', 'rhythm', 'kick', 'snare'],
            'vocal': ['vocal', 'voice', 'speech', 'talk', 'narr', 'dialogue'],
            'nature': ['nature', 'wind', 'rain', 'water', 'forest', 'bird'],
            'mechanical': ['mech', 'robot', 'machine', 'tech', 'digital', 'synth'],
            'emotional': ['sad', 'happy', 'dark', 'bright', 'calm', 'tense'],
            'temporal': ['intro', 'outro', 'loop', 'oneshot', 'sustained']
        }
        
        for category, keywords in descriptors.items():
            if any(keyword in filename_lower for keyword in keywords):
                patterns.append(category)
        
        return patterns
    
    def build_adaptive_prompt(self, file_path, metadata):
        """Build a prompt that learns from previous classifications"""
        
        # Analyze filename patterns
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        
        # Get historical context
        similar_files = self.find_similar_files(file_path.name)
        
        # Build dynamic categories list
        all_categories = list(self.base_categories.keys()) + self.discovered_categories.get('new_categories', [])
        all_moods = []
        for cat_moods in self.base_categories.values():
            all_moods.extend(cat_moods)
        all_moods.extend(self.discovered_categories.get('new_moods', []))
        all_moods = list(set(all_moods))  # Remove duplicates
        
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        context = ""
        if similar_files:
            context = f"\nCONTEXT from similar files:\n"
            for similar in similar_files[:3]:
                context += f"- {similar['filename']}: {similar['category']} + {similar['mood']}\n"
        
        if filename_patterns:
            context += f"\nFILENAME PATTERNS detected: {', '.join(filename_patterns)}\n"
        
        prompt = f"""You are an expert audio librarian with adaptive learning capabilities. Analyze this audio file and classify it, considering both standard categories and potentially discovering new ones.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
{context}

CLASSIFICATION FRAMEWORK:

CONTENT CATEGORIES (choose existing or suggest new):
Existing: {', '.join(all_categories)}
- If this doesn't fit existing categories, suggest a new category in format: category_newname

PRIMARY MOODS (choose existing or suggest new):
Existing: {', '.join(all_moods)}
- If this doesn't fit existing moods, suggest a new mood

INTENSITY LEVELS:
- subtle_background: Unobtrusive, atmospheric support
- narrative_support: Clear presence but supports story
- dramatic_punctuation: Bold, attention-grabbing moments

LEARNING INSTRUCTIONS:
- Be creative and specific in your classifications
- If you detect patterns that don't fit existing categories, suggest new ones
- Consider the creative context of AI consciousness storytelling
- Use filename patterns as clues but don't be limited by them

Return response as JSON:
{{
  "category": "existing_category_or_new_category",
  "mood": "existing_mood_or_new_mood",
  "intensity": "subtle_background|narrative_support|dramatic_punctuation",
  "energy_level": 1-10,
  "tags": ["tag1", "tag2", "tag3"],
  "thematic_notes": "How this could enhance AI consciousness storytelling",
  "suggested_filename": "descriptive_filename",
  "confidence": 0.0-1.0,
  "reasoning": "Why you chose these classifications",
  "discovered_elements": ["any new patterns or categories you noticed"]
}}"""
        
        return prompt
    
    def find_similar_files(self, filename):
        """Find similar files in learning data"""
        similar = []
        for entry in self.learning_data['classifications']:
            if self.filename_similarity(filename, entry['filename']) > 0.3:
                similar.append(entry)
        return similar[:5]  # Return top 5 similar files
    
    def filename_similarity(self, filename1, filename2):
        """Calculate similarity between filenames"""
        # Simple word-based similarity
        words1 = set(filename1.lower().replace('_', ' ').split())
        words2 = set(filename2.lower().replace('_', ' ').split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def learn_from_classification(self, file_path, classification):
        """Learn from each classification to improve future ones"""
        
        # Store classification
        learning_entry = {
            'filename': file_path.name,
            'classification': classification,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        
        self.learning_data['classifications'].append(learning_entry)
        
        # Track new categories/moods
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        
        # Check for new categories
        if category not in [cat for cats in self.base_categories.keys() for cat in cats]:
            if category not in self.discovered_categories['new_categories']:
                self.discovered_categories['new_categories'].append(category)
                print(f"🆕 Discovered new category: {category}")
        
        # Check for new moods
        all_base_moods = [mood for moods in self.base_categories.values() for mood in moods]
        if mood not in all_base_moods:
            if mood not in self.discovered_categories['new_moods']:
                self.discovered_categories['new_moods'].append(mood)
                print(f"🆕 Discovered new mood: {mood}")
        
        # Update frequency counts
        self.discovered_categories['frequency_counts'][f"{category}+{mood}"] += 1
        
        # Learn filename patterns
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        for pattern in filename_patterns:
            self.learning_data['filename_patterns'][pattern].append({
                'category': category,
                'mood': mood,
                'filename': file_path.name
            })
        
        # Save learning data
        self.save_learning_data()
        self.save_discovered_categories()
        
        # Rebuild folder map with new discoveries
        self.folder_map = self.build_dynamic_folder_map()
    
    def classify_audio_file(self, file_path, user_description=""):
        """Classify with adaptive learning"""
        
        metadata = self.get_audio_metadata(file_path)
        prompt = self.build_adaptive_prompt(file_path, metadata)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4  # Slightly higher for more creativity
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # Handle markdown-wrapped JSON
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(raw_response)
            
            # Learn from this classification
            self.learn_from_classification(file_path, classification)
            
            return classification
            
        except Exception as e:
            print(f"Classification failed: {e}")
            return None
    
    def determine_target_folder(self, classification):
        """Determine target folder, creating new ones if needed"""
        if not classification:
            return self.folder_map["default"]
        
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        intensity = classification.get('intensity', '')
        
        # Try exact match first
        key = f"{category} + {mood}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Try with intensity
        key = f"{category} + {intensity}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Create new folder path for discovered categories
        if category.startswith('music_'):
            new_path = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '')
            new_path = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
        elif category.startswith('voice_'):
            new_path = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
        else:
            new_path = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        # Add to folder map
        self.folder_map[key] = new_path
        print(f"🆕 Created new folder mapping: {key} → {new_path}")
        
        return new_path
    
    def process_file(self, file_path, user_description="", dry_run=True):
        """Process file with adaptive learning"""
        print(f"\n🔍 Processing: {file_path.name}")
        
        classification = self.classify_audio_file(file_path, user_description)
        
        if classification:
            target_folder = self.determine_target_folder(classification)
            
            print(f"  📂 Category: {classification.get('category', 'unknown')}")
            print(f"  🎭 Mood: {classification.get('mood', 'unknown')}")
            print(f"  ⚡ Intensity: {classification.get('intensity', 'unknown')}")
            print(f"  🔥 Energy: {classification.get('energy_level', 0)}/10")
            print(f"  🎯 Confidence: {classification.get('confidence', 0):.1%}")
            print(f"  📍 Target: {target_folder}")
            print(f"  🏷️ Tags: {', '.join(classification.get('tags', []))}")
            
            if classification.get('discovered_elements'):
                print(f"  🆕 Discovered: {', '.join(classification.get('discovered_elements', []))}")
            
            if not dry_run:
                # Would move file here
                print(f"  ✅ Would move to: {target_folder}")
            else:
                print(f"  🔄 [DRY RUN] Would move to: {target_folder}")
            
            return classification
        else:
            print(f"  ❌ Classification failed")
            return None
    
    def show_learning_stats(self):
        """Display learning statistics"""
        print(f"\n📊 LEARNING STATISTICS")
        print(f"=" * 50)
        
        total_files = len(self.learning_data['classifications'])
        print(f"Total files processed: {total_files}")
        
        print(f"\n🆕 DISCOVERED CATEGORIES:")
        for category in self.discovered_categories.get('new_categories', []):
            print(f"  - {category}")
        
        print(f"\n🆕 DISCOVERED MOODS:")
        for mood in self.discovered_categories.get('new_moods', []):
            print(f"  - {mood}")
        
        print(f"\n📈 MOST COMMON COMBINATIONS:")
        freq_counts = self.discovered_categories.get('frequency_counts', {})
        for combo, count in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {combo}: {count} files")
        
        print(f"\n🔍 FILENAME PATTERNS LEARNED:")
        for pattern, examples in list(self.learning_data['filename_patterns'].items())[:5]:
            print(f"  - {pattern}: {len(examples)} examples")

# Usage example
if __name__ == "__main__":
    try:
        from config import OPENAI_API_KEY, BASE_DIRECTORY, ensure_base_structure
        ensure_base_structure()
        
        # Initialize adaptive organizer
        organizer = AdaptiveAudioOrganizer(OPENAI_API_KEY, str(BASE_DIRECTORY))
    except RuntimeError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
    
    # Show current learning stats
    organizer.show_learning_stats()
    
    # Process files - system will learn and adapt
    # organizer.process_file(Path("test.mp3"), dry_run=True)

# Imports and setup
import os
import shutil
import json
import time
from pathlib import Path
from openai import OpenAI
import mutagen
from mutagen.id3 import ID3NoHeaderError

# Cell 3: Configuration 
try:
    from config import OPENAI_API_KEY, BASE_DIRECTORY, ensure_base_structure
    ensure_base_structure()
except RuntimeError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)

# Import directories to scan from config, with fallback to base directory
from config import DIRECTORIES_TO_SCAN
if not DIRECTORIES_TO_SCAN:
    DIRECTORIES_TO_SCAN = [BASE_DIRECTORY]
# Cell 4: The AudioOrganizer class
class AudioOrganizer:
    def __init__(self, openai_api_key, base_directory):
        self.client = OpenAI(api_key=openai_api_key)
        self.base_dir = Path(base_directory)
        
        # Your folder structure mapping
        self.folder_map = {
            "music_ambient + contemplative": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/contemplative/",
            "music_ambient + tension_building": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/tension_building/",
            "music_ambient + wonder_discovery": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/wonder_discovery/",
            "music_ambient + melancholic": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/melancholic/",
            "music_ambient + mysterious": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/mysterious/",
            "sfx_consciousness + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/thought_processing/",
            "sfx_consciousness + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/awakening_emergence/",
            "sfx_consciousness + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/memory_formation/",
            "sfx_human + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/breathing_heartbeat/",
            "sfx_human + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/emotional_responses/",
            "sfx_human + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/environmental_human/",
            "sfx_environmental + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/time_space/",
            "sfx_environmental + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/transformation/",
            "sfx_environmental + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/connection_bridging/",
            "voice_element + contemplative": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/narrator_banks/",
            "voice_element + melancholic": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/processed_vocals/",
            "voice_element + mysterious": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/vocal_textures/",
            "voice_element + dramatic_punctuation": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/character_voices/",
            "theme_human_machine_dialogue": "THEMATIC_COLLECTIONS/human_machine_dialogue/",
            "theme_emergence_awakening": "THEMATIC_COLLECTIONS/emergence_awakening/",
            "theme_digital_consciousness": "THEMATIC_COLLECTIONS/digital_consciousness/",
            "theme_memory_formation": "THEMATIC_COLLECTIONS/memory_formation/",
            "theme_ethical_questions": "THEMATIC_COLLECTIONS/ethical_questions/",
            "theme_connection_bridge_sounds": "THEMATIC_COLLECTIONS/connection_bridge_sounds/",
            "default": "01_UNIVERSAL_ASSETS/UNSORTED/"
        }
        
        self.audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
        
    def get_audio_metadata(self, file_path):
        """Extract audio metadata using mutagen"""
        try:
            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                duration = audio_file.info.length
                return {
                    'duration': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'bitrate': getattr(audio_file.info, 'bitrate', 'Unknown'),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 'Unknown')
                }
        except Exception as e:
            print(f"Could not read metadata for {file_path}: {e}")
        return {'duration': 'Unknown', 'bitrate': 'Unknown', 'sample_rate': 'Unknown'}
    
    def classify_audio_file(self, file_path, user_description=""):
        """Send file info to OpenAI for classification"""
        
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        metadata = self.get_audio_metadata(file_path)
        
        prompt = f"""You are an expert audio librarian specializing in storytelling and podcast production. Analyze this audio asset for a creative series about AI consciousness and human-AI relationships.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
- Description: {user_description}

CLASSIFICATION FRAMEWORK:

PRIMARY MOOD (choose one):
- contemplative: Reflective, thoughtful, introspective
- tension_building: Suspenseful, growing unease, anticipation
- wonder_discovery: Awe, curiosity, magical realization
- melancholic: Bittersweet, nostalgic, gentle sadness
- mysterious: Enigmatic, unknown, hidden depths

INTENSITY LEVEL (choose one):
- subtle_background: Unobtrusive, atmospheric support
- narrative_support: Clear presence but supports story
- dramatic_punctuation: Bold, attention-grabbing moments

CONTENT CATEGORY (choose one):
- music_ambient: Musical pieces, melodies, harmonies
 - sfx_technology: Computer sounds, data, mechanical
- sfx_consciousness: Abstract sounds suggesting thought/awareness
- sfx_human: Breathing, heartbeat, emotional responses
- sfx_environmental: Nature, spaces, atmospheric
- voice_element: Processed vocals, speech, vocal textures

Return response as JSON:
{{
  "mood": "",
  "intensity": "",
  "category": "",
  "energy_level": 0,
  "tags": [],
  "thematic_notes": "",
  "suggested_filename": "",
  "cross_reference_folders": []
}}"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            classification = json.loads(response.choices[0].message.content)
            return classification
            
        except Exception as e:
            print(f"OpenAI classification failed for {file_path}: {e}")
            return None
    
    def determine_target_folder(self, classification):
        """Map classification to target folder"""
        if not classification:
            return self.folder_map["default"]
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        intensity = classification.get('intensity', '')
        
        # Create the key for folder mapping
        key = f"{category} + {mood}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Try with intensity
        key = f"{category} + {intensity}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        return self.folder_map["default"]
    
    def scan_directory(self, scan_path):
        """Find all audio files in directory"""

        audio_files = []
        scan_path = Path(scan_path)
        
        for file_path in scan_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.audio_extensions:
                audio_files.append(file_path)
        
        return audio_files
    
    def process_file(self, file_path, user_description="", dry_run=True):
        """Process a single audio file"""
        print(f"\nProcessing: {file_path.name}")
        
        # Get classification from OpenAI
        classification = self.classify_audio_file(file_path, user_description)
        
        if classification:
            target_folder = self.determine_target_folder(classification)
            
            print(f"  Category: {classification.get('category', 'unknown')}")
            print(f"  Mood: {classification.get('mood', 'unknown')}")
            print(f"  Target: {target_folder}")
            print(f"  Tags: {', '.join(classification.get('tags', []))}")
            
            if not dry_run:
                # Would actually move file here
                print(f"  ✓ Would move to: {target_folder}")
            else:
                print(f"  [DRY RUN] Would move to: {target_folder}")
                
            return classification
        else:
            print(f"  ✗ Classification failed")
            return None
            
    

# Cell 5: Initialize the organizer
organizer = AudioOrganizer(OPENAI_API_KEY, BASE_DIRECTORY)

# Cell: Enhanced classification with both approaches
def enhanced_classify_audio_file(self, file_path, user_description=""):
    """Classify using BOTH filename patterns AND actual audio analysis"""
    
    metadata = self.get_audio_metadata(file_path)
    
    # Analyze actual audio content
    audio_analysis = self.analyze_audio_content(file_path)
    
    # Get filename patterns
    filename_patterns = self.analyze_filename_patterns(file_path.name)
    similar_files = self.find_similar_files(file_path.name)
    
    file_stats = os.stat(file_path)
    file_size_mb = file_stats.st_size / (1024 * 1024)
    
    # Build comprehensive context
    context = ""
    
    if similar_files:
        context += f"\nSIMILAR FILES:\n"
        for similar in similar_files[:3]:
            context += f"- {similar['filename']}: {similar['category']} + {similar['mood']}\n"
    
    if filename_patterns:
        context += f"\nFILENAME PATTERNS: {', '.join(filename_patterns)}\n"
    
    # Add audio analysis insights
    if audio_analysis:
        interp = audio_analysis['interpretation']
        raw = audio_analysis['raw_features']
        context += f"\nAUDIO ANALYSIS:\n"
        context += f"- Brightness: {interp.get('brightness', 'unknown')}\n"
        context += f"- Texture: {interp.get('texture', 'unknown')}\n"
        context += f"- Energy: {interp.get('energy', 'unknown')}\n"
        context += f"- Pace: {interp.get('pace', 'unknown')}\n"
        context += f"- Rhythmic Activity: {interp.get('rhythmic_activity', 'unknown')}\n"
        context += f"- Tempo: {raw.get('tempo', 0):.1f} BPM\n"
    
    # Get dynamic categories
    all_categories = list(self.base_categories.keys()) + self.discovered_categories.get('new_categories', [])
    all_moods = []
    for cat_moods in self.base_categories.values():
        all_moods.extend(cat_moods)
    all_moods.extend(self.discovered_categories.get('new_moods', []))
    all_moods = list(set(all_moods))
    
    prompt = f"""You are an expert audio librarian with access to both filename analysis AND actual audio content analysis. Use this combined information for the most accurate classification.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
{context}

CLASSIFICATION FRAMEWORK:
Available categories: {', '.join(all_categories)}
Available moods: {', '.join(all_moods)}

INTENSITY LEVELS:
- subtle_background: Unobtrusive, atmospheric support
- narrative_support: Clear presence but supports story  
- dramatic_punctuation: Bold, attention-grabbing moments

INSTRUCTIONS:
- Combine filename clues with actual audio characteristics
- Audio analysis should take precedence over filename when they conflict
- Suggest new categories/moods if the audio doesn't fit existing ones
- Explain how both filename and audio analysis support your decision

Return JSON:
{{
  "category": "category_name",
  "mood": "mood_name", 
  "intensity": "intensity_level",
  "energy_level": 1-10,
  "tags": ["tag1", "tag2"],
  "thematic_notes": "How this enhances AI consciousness storytelling",
  "suggested_filename": "descriptive_name",
  "confidence": 0.0-1.0,
  "reasoning": "Why you chose this, combining filename and audio insights",
  "discovered_elements": ["new patterns found"],
  "audio_insights": "What the audio analysis revealed"
}}"""
    
    try:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        raw_response = response.choices[0].message.content.strip()
        
        if raw_response.startswith('```json'):
            raw_response = raw_response.replace('```json', '').replace('```', '').strip()
        
        classification = json.loads(raw_response)
        
        # Store audio analysis with classification
        if audio_analysis:
            classification['audio_analysis'] = audio_analysis
        
        self.learn_from_classification(file_path, classification)
        return classification
        
    except Exception as e:
        print(f"Enhanced classification failed: {e}")
        return None

# Replace the method
AdaptiveAudioOrganizer.classify_audio_file = enhanced_classify_audio_file

print("🎵 Enhanced classification ready!")
print("Now using BOTH filename patterns AND actual audio analysis!")
# Cell: Full recursive system sweep with enhanced audio analysis
print("🚀 Starting FULL SYSTEM SWEEP with enhanced audio analysis!")
print("This will:")
print("- 🎵 Analyze actual audio content (tempo, brightness, energy, etc.)")
print("- 📝 Learn from filename patterns")
print("- 🧠 Build knowledge base from your entire library")
print("- 📊 Discover new categories and patterns")
print("=" * 70)

# Configure for full sweep
DIRECTORIES_TO_SCAN = [
    str(BASE_DIRECTORY),  # Everything!
]

# Find ALL audio files recursively
print("🔍 Scanning for audio files...")
all_audio_files = []

for scan_path in DIRECTORIES_TO_SCAN:
    scan_path = Path(scan_path)
    if scan_path.exists():
        for file_path in scan_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}:
                # Skip files already in organized folders to avoid re-processing
                if not any(skip_folder in str(file_path) for skip_folder in ['01_UNIVERSAL_ASSETS', 'THEMATIC_COLLECTIONS']):
                    all_audio_files.append(file_path)

print(f"📊 Found {len(all_audio_files)} audio files to process")

# Show what we found
folders_found = {}
for file_path in all_audio_files:
    folder = str(file_path.parent.relative_to(BASE_DIRECTORY))
    folders_found[folder] = folders_found.get(folder, 0) + 1

print(f"\n📂 Files by folder:")
for folder, count in sorted(folders_found.items()):
    print(f"  {folder}: {count} files")

if len(all_audio_files) == 0:
    print("❌ No files found to process!")
else:
    print(f"\n🎯 Ready to process {len(all_audio_files)} files with enhanced analysis")
    print("💡 This will take some time due to audio analysis...")
# Cell: Create the enhanced organizer instance
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    exit(1)
BASE_DIRECTORY = os.getenv('AUDIOAI_BASE_DIRECTORY')
if not BASE_DIRECTORY:
    print("❌ Please set AUDIOAI_BASE_DIRECTORY environment variable")
    print("   export AUDIOAI_BASE_DIRECTORY='/path/to/your/audio/library'")
    exit(1) 
# Create the enhanced organizer with all the new methods
organizer = AdaptiveAudioOrganizer(OPENAI_API_KEY, BASE_DIRECTORY)

print("✅ Enhanced organizer created!")
print("🎵 Audio analysis: Ready")
print("🏷️ BPM filename generation: Ready")
print("🧠 Adaptive learning: Ready")
# Cell: Complete Enhanced AdaptiveAudioOrganizer
import os
import shutil
import json
import time
from pathlib import Path
from openai import OpenAI
import mutagen
from collections import defaultdict
import pickle
from datetime import datetime
import librosa
import numpy as np
import hashlib

class AdaptiveAudioOrganizer:
    def __init__(self, openai_api_key, base_directory):
        self.client = OpenAI(api_key=openai_api_key)
        self.base_dir = Path(base_directory)
        
        # Learning system files
        self.learning_data_file = self.base_dir / "04_METADATA_SYSTEM" / "learning_data.pkl"
        self.discovered_categories_file = self.base_dir / "04_METADATA_SYSTEM" / "discovered_categories.json"
        
        # Load existing learning data
        self.learning_data = self.load_learning_data()
        self.discovered_categories = self.load_discovered_categories()
        
        # Base categories that can expand
        self.base_categories = {
            "music_ambient": ["contemplative", "tension_building", "wonder_discovery", "melancholic", "mysterious"],
            "sfx_consciousness": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_human": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_environmental": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_technology": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "voice_element": ["contemplative", "melancholic", "mysterious", "dramatic_punctuation"],
        }
        
        # Dynamic folder mapping
        self.folder_map = self.build_dynamic_folder_map()
        self.audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
        
    def load_learning_data(self):
        """Load historical classification data"""
        if self.learning_data_file.exists():
            with open(self.learning_data_file, 'rb') as f:
                return pickle.load(f)
        return {
            'classifications': [],
            'user_corrections': [],
            'patterns': defaultdict(list),
            'filename_patterns': defaultdict(list)
        }
    
    def save_learning_data(self):
        """Save learning data for future use"""
        self.learning_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.learning_data_file, 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def load_discovered_categories(self):
        """Load dynamically discovered categories"""
        if self.discovered_categories_file.exists():
            with open(self.discovered_categories_file, 'r') as f:
                return json.load(f)
        return {
            'new_moods': [],
            'new_categories': [],
            'new_themes': [],
            'frequency_counts': defaultdict(int)
        }
    
    def save_discovered_categories(self):
        """Save discovered categories"""
        self.discovered_categories_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.discovered_categories_file, 'w') as f:
            json.dump(self.discovered_categories, f, indent=2)
    
    def build_dynamic_folder_map(self):
        """Build folder mapping that includes discovered categories"""
        base_map = {
            "music_ambient + contemplative": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/contemplative/",
            "music_ambient + tension_building": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/tension_building/",
            "music_ambient + wonder_discovery": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/wonder_discovery/",
            "music_ambient + melancholic": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/melancholic/",
            "music_ambient + mysterious": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/mysterious/",
            "sfx_consciousness + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/thought_processing/",
            "sfx_consciousness + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/awakening_emergence/",
            "sfx_consciousness + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/memory_formation/",
            "sfx_human + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/breathing_heartbeat/",
            "sfx_human + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/emotional_responses/",
            "sfx_human + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/environmental_human/",
            "sfx_environmental + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/time_space/",
            "sfx_environmental + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/transformation/",
            "sfx_environmental + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/connection_bridging/",
            "voice_element + contemplative": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/narrator_banks/",
            "voice_element + melancholic": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/processed_vocals/",
            "voice_element + mysterious": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/vocal_textures/",
            "voice_element + dramatic_punctuation": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/character_voices/",
            "default": "01_UNIVERSAL_ASSETS/UNSORTED/"
        }
        
        # Add discovered categories
        for category in self.discovered_categories.get('new_categories', []):
            for mood in self.discovered_categories.get('new_moods', []):
                key = f"{category} + {mood}"
                if key not in base_map:
                    if category.startswith('music_'):
                        base_map[key] = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
                    elif category.startswith('sfx_'):
                        sfx_type = category.replace('sfx_', '')
                        base_map[key] = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
                    elif category.startswith('voice_'):
                        base_map[key] = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
                    else:
                        base_map[key] = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        return base_map
    
    def get_audio_metadata(self, file_path):
        """Extract audio metadata using mutagen"""
        try:
            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                duration = audio_file.info.length
                return {
                    'duration': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'bitrate': getattr(audio_file.info, 'bitrate', 'Unknown'),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 'Unknown')
                }
        except Exception as e:
            print(f"Could not read metadata for {file_path}: {e}")
        return {'duration': 'Unknown', 'bitrate': 'Unknown', 'sample_rate': 'Unknown'}
    
    def analyze_audio_content(self, file_path):
        """Actually analyze the audio content"""
        try:
            print(f"  🎵 Analyzing audio content...")
            
            # Load first 30 seconds (faster processing)
            y, sr = librosa.load(file_path, duration=30)
            
            features = {}
            
            # Spectral brightness
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate (noisiness vs tonality)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Tempo analysis
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy'] = float(np.mean(rms))
            
            # Onset detection (rhythmic activity)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_rate'] = len(onset_frames) / (len(y) / sr)
            
            # Spectral contrast (timbral texture)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            
            # Interpret features into human-readable descriptions
            interpretation = self.interpret_audio_features(features)
            
            return {
                'raw_features': features,
                'interpretation': interpretation
            }
            
        except Exception as e:
            print(f"  ⚠️  Audio analysis failed: {e}")
            return None

    def interpret_audio_features(self, features):
        """Convert raw features to human-readable descriptions"""
        interpretation = {}
        
        # Brightness (spectral centroid)
        if features['spectral_centroid_mean'] > 3000:
            interpretation['brightness'] = 'bright'
        elif features['spectral_centroid_mean'] > 1000:
            interpretation['brightness'] = 'balanced'
        else:
            interpretation['brightness'] = 'dark'
        
        # Texture (zero crossing rate)
        if features['zero_crossing_rate'] > 0.1:
            interpretation['texture'] = 'noisy'
        elif features['zero_crossing_rate'] > 0.05:
            interpretation['texture'] = 'moderate'
        else:
            interpretation['texture'] = 'tonal'
        
        # Energy level
        if features['rms_energy'] > 0.1:
            interpretation['energy'] = 'high'
        elif features['rms_energy'] > 0.05:
            interpretation['energy'] = 'medium'
        else:
            interpretation['energy'] = 'low'
        
        # Tempo interpretation
        if features['tempo'] > 120:
            interpretation['pace'] = 'fast'
        elif features['tempo'] > 80:
            interpretation['pace'] = 'moderate'
        else:
            interpretation['pace'] = 'slow'
        
        # Rhythmic activity
        if features['onset_rate'] > 2:
            interpretation['rhythmic_activity'] = 'high'
        elif features['onset_rate'] > 0.5:
            interpretation['rhythmic_activity'] = 'moderate'
        else:
            interpretation['rhythmic_activity'] = 'low'
        
        return interpretation

    def analyze_filename_patterns(self, filename):
        """Extract patterns from filename that might indicate content type"""
        patterns = []
        filename_lower = filename.lower()
        
        # Common audio descriptors
        descriptors = {
            'ambient': ['ambient', 'atmosphere', 'atmos', 'background', 'bg'],
            'percussion': ['drum', 'beat', 'perc', 'rhythm', 'kick', 'snare'],
            'vocal': ['vocal', 'voice', 'speech', 'talk', 'narr', 'dialogue'],
            'nature': ['nature', 'wind', 'rain', 'water', 'forest', 'bird'],
            'mechanical': ['mech', 'robot', 'machine', 'tech', 'digital', 'synth'],
            'emotional': ['sad', 'happy', 'dark', 'bright', 'calm', 'tense'],
            'temporal': ['intro', 'outro', 'loop', 'oneshot', 'sustained']
        }
        
        for category, keywords in descriptors.items():
            if any(keyword in filename_lower for keyword in keywords):
                patterns.append(category)
        
        return patterns
    
    def find_similar_files(self, filename):
        """Find similar files in learning data"""
        similar = []
        for entry in self.learning_data['classifications']:
            if self.filename_similarity(filename, entry['filename']) > 0.3:
                similar.append(entry)
        return similar[:5]
    
    def filename_similarity(self, filename1, filename2):
        """Calculate similarity between filenames"""
        words1 = set(filename1.lower().replace('_', ' ').split())
        words2 = set(filename2.lower().replace('_', ' ').split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def generate_enhanced_filename(self, file_path, classification, audio_analysis):
        """Generate descriptive filename including BPM and other useful info"""
        
        # Extract key info
        category = classification.get('category', 'unknown')
        mood = classification.get('mood', 'unknown')
        intensity = classification.get('intensity', 'unknown')
        energy = classification.get('energy_level', 0)
        
        # Get audio characteristics
        bpm = None
        brightness = None
        texture = None
        
        if audio_analysis:
            raw_features = audio_analysis.get('raw_features', {})
            interpretation = audio_analysis.get('interpretation', {})
            
            bpm = raw_features.get('tempo', 0)
            brightness = interpretation.get('brightness', '')
            texture = interpretation.get('texture', '')
        
        # Build filename components
        components = []
        
        # 1. Category prefix
        if category.startswith('music_'):
            components.append('MUS')
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '').upper()
            components.append(f'SFX_{sfx_type[:4]}')
        elif category.startswith('voice_'):
            components.append('VOX')
        else:
            components.append(category.upper()[:6])
        
        # 2. BPM (if detected and makes sense)
        if bpm and bpm > 30 and bpm < 300:
            components.append(f"{int(bpm)}bpm")
        
        # 3. Mood/Energy
        mood_short = mood[:4].upper() if mood else ""
        if mood_short:
            components.append(mood_short)
        
        # 4. Energy level
        if energy:
            components.append(f"E{energy}")
        
        # 5. Audio characteristics
        if brightness in ['bright', 'dark']:
            components.append(brightness[:3].upper())
        
        if texture == 'tonal':
            components.append('TON')
        elif texture == 'noisy':
            components.append('NOI')
        
        # 6. Intensity indicator
        intensity_map = {
            'subtle_background': 'BG',
            'narrative_support': 'SUP', 
            'dramatic_punctuation': 'DRA'
        }
        if intensity in intensity_map:
            components.append(intensity_map[intensity])
        
        # Join components
        new_filename = '_'.join(components)
        
        # Keep original extension
        original_ext = file_path.suffix
        
        # Add uniqueness
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:4]
        
        final_filename = f"{new_filename}_{file_hash}{original_ext}"
        
        return final_filename
    
    def classify_audio_file(self, file_path, user_description=""):
        """Classify using BOTH filename patterns AND actual audio analysis"""
        
        metadata = self.get_audio_metadata(file_path)
        
        # Analyze actual audio content
        audio_analysis = self.analyze_audio_content(file_path)
        
        # Get filename patterns
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        similar_files = self.find_similar_files(file_path.name)
        
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Build comprehensive context
        context = ""
        
        if similar_files:
            context += f"\nSIMILAR FILES:\n"
            for similar in similar_files[:3]:
                context += f"- {similar['filename']}: {similar['category']} + {similar['mood']}\n"
        
        if filename_patterns:
            context += f"\nFILENAME PATTERNS: {', '.join(filename_patterns)}\n"
        
        # Add audio analysis insights
        if audio_analysis:
            interp = audio_analysis['interpretation']
            raw = audio_analysis['raw_features']
            context += f"\nAUDIO ANALYSIS:\n"
            context += f"- Brightness: {interp.get('brightness', 'unknown')}\n"
            context += f"- Texture: {interp.get('texture', 'unknown')}\n"
            context += f"- Energy: {interp.get('energy', 'unknown')}\n"
            context += f"- Pace: {interp.get('pace', 'unknown')}\n"
            context += f"- Rhythmic Activity: {interp.get('rhythmic_activity', 'unknown')}\n"
            context += f"- Tempo: {raw.get('tempo', 0):.1f} BPM\n"
        
        # Get dynamic categories
        all_categories = list(self.base_categories.keys()) + self.discovered_categories.get('new_categories', [])
        all_moods = []
        for cat_moods in self.base_categories.values():
            all_moods.extend(cat_moods)
        all_moods.extend(self.discovered_categories.get('new_moods', []))
        all_moods = list(set(all_moods))
        
        prompt = f"""You are an expert audio librarian with access to both filename analysis AND actual audio content analysis. Use this combined information for the most accurate classification.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
{context}

CLASSIFICATION FRAMEWORK:
Available categories: {', '.join(all_categories)}
Available moods: {', '.join(all_moods)}

INSTRUCTIONS:
- Combine filename clues with actual audio characteristics
- Audio analysis should take precedence over filename when they conflict
- Suggest new categories/moods if the audio doesn't fit existing ones
- Be precise about BPM if detected - this will be used in the filename

Return JSON:
{{
  "category": "category_name",
  "mood": "mood_name", 
  "intensity": "intensity_level",
  "energy_level": 1-10,
  "tags": ["tag1", "tag2"],
  "thematic_notes": "How this enhances AI consciousness storytelling",
  "confidence": 0.0-1.0,
  "reasoning": "Why you chose this, combining filename and audio insights",
  "discovered_elements": ["new patterns found"],
  "audio_insights": "What the audio analysis revealed"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(raw_response)
            
            # Generate enhanced filename with BPM
            enhanced_filename = self.generate_enhanced_filename(file_path, classification, audio_analysis)
            classification['suggested_filename'] = enhanced_filename
            
            # Store audio analysis
            if audio_analysis:
                classification['audio_analysis'] = audio_analysis
            
            self.learn_from_classification(file_path, classification)
            return classification
            
        except Exception as e:
            print(f"Enhanced classification failed: {e}")
            return None
    
    def learn_from_classification(self, file_path, classification):
        """Learn from each classification to improve future ones"""
        
        learning_entry = {
            'filename': file_path.name,
            'classification': classification,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        
        self.learning_data['classifications'].append(learning_entry)
        
        # Track new categories/moods
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        
        # Check for new categories
        if category not in [cat for cats in self.base_categories.keys() for cat in cats]:
            if category not in self.discovered_categories['new_categories']:
                self.discovered_categories['new_categories'].append(category)
                print(f"🆕 Discovered new category: {category}")
        
        # Check for new moods
        all_base_moods = [mood for moods in self.base_categories.values() for mood in moods]
        if mood not in all_base_moods:
            if mood not in self.discovered_categories['new_moods']:
                self.discovered_categories['new_moods'].append(mood)
                print(f"🆕 Discovered new mood: {mood}")
        
        # Update frequency counts
        self.discovered_categories['frequency_counts'][f"{category}+{mood}"] += 1
        
        # Learn filename patterns
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        for pattern in filename_patterns:
            self.learning_data['filename_patterns'][pattern].append({
                'category': category,
                'mood': mood,
                'filename': file_path.name
            })
        
        # Save learning data
        self.save_learning_data()
        self.save_discovered_categories()
        
        # Rebuild folder map with new discoveries
        self.folder_map = self.build_dynamic_folder_map()
    
    def determine_target_folder(self, classification):
        """Determine target folder, creating new ones if needed"""
        if not classification:
            return self.folder_map["default"]
        
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        intensity = classification.get('intensity', '')
        
        # Try exact match first
        key = f"{category} + {mood}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Try with intensity
        key = f"{category} + {intensity}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Create new folder path for discovered categories
        if category.startswith('music_'):
            new_path = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '')
            new_path = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
        elif category.startswith('voice_'):
            new_path = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
        else:
            new_path = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        # Add to folder map
        self.folder_map[key] = new_path
        print(f"🆕 Created new folder mapping: {key} → {new_path}")
        
        return new_path
    
    def process_file(self, file_path, user_description="", dry_run=True):
        """Process file with adaptive learning"""
        print(f"\n🔍 Processing: {file_path.name}")
        
        classification = self.classify_audio_file(file_path, user_description)
        
        if classification:
            target_folder = self.determine_target_folder(classification)
            
            print(f"  📂 Category: {classification.get('category', 'unknown')}")
            print(f"  🎭 Mood: {classification.get('mood', 'unknown')}")
            print(f"  ⚡ Intensity: {classification.get('intensity', 'unknown')}")
            print(f"  🔥 Energy: {classification.get('energy_level', 0)}/10")
            print(f"  🎯 Confidence: {classification.get('confidence', 0):.1%}")
            print(f"  📍 Target: {target_folder}")
            print(f"  🏷️ Enhanced filename: {classification.get('suggested_filename', 'None')}")
            print(f"  🏷️ Tags: {', '.join(classification.get('tags', []))}")
            
            if classification.get('discovered_elements'):
                print(f"  🆕 Discovered: {', '.join(classification.get('discovered_elements', []))}")
            
            if not dry_run:
                print(f"  ✅ Would move to: {target_folder}")
            else:
                print(f"  🔄 [DRY RUN] Would move to: {target_folder}")
            
            return classification
        else:
            print(f"  ❌ Classification failed")
            return None
    
    def show_learning_stats(self):
        """Display learning statistics"""
        print(f"\n📊 LEARNING STATISTICS")
        print(f"=" * 50)
        
        total_files = len(self.learning_data['classifications'])
        print(f"Total files processed: {total_files}")
        
        print(f"\n🆕 DISCOVERED CATEGORIES:")
        for category in self.discovered_categories.get('new_categories', []):
            print(f"  - {category}")
        
        print(f"\n🆕 DISCOVERED MOODS:")
        for mood in self.discovered_categories.get('new_moods', []):
            print(f"  - {mood}")
        
        print(f"\n📈 MOST COMMON COMBINATIONS:")
        freq_counts = self.discovered_categories.get('frequency_counts', {})
        for combo, count in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {combo}: {count} files")

print("🎵 Complete Enhanced AdaptiveAudioOrganizer created!")
print("✅ Includes: Audio analysis, BPM detection, adaptive learning, enhanced filenames")
# Cell: Test with your file
from config import resolve_test_file
test_file = resolve_test_file()

if test_file and test_file.exists():
    result = organizer.process_file(Path(test_file), dry_run=True)
    print("🎉 Test successful!")
else:
    print("❌ Test file not found")
# Cell: Fix tempo conversion
def fixed_analyze_audio_content(self, file_path):
    """Actually analyze the audio content - FIXED VERSION"""
    try:
        print(f"  🎵 Analyzing audio content...")
        
        # Load first 30 seconds (faster processing)
        y, sr = librosa.load(file_path, duration=30)
        
        features = {}
        
        # Spectral brightness
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        # Spectral rolloff (high frequency content)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Zero crossing rate (noisiness vs tonality)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate'] = float(np.mean(zcr))
        
        # Tempo analysis - FIXED
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy'] = float(np.mean(rms))
        
        # Onset detection (rhythmic activity)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        features['onset_rate'] = len(onset_frames) / (len(y) / sr)
        
        # Spectral contrast (timbral texture)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(contrast))
        
        # Interpret features into human-readable descriptions
        interpretation = self.interpret_audio_features(features)
        
        return {
            'raw_features': features,
            'interpretation': interpretation
        }
        
    except Exception as e:
        print(f"  ⚠️  Audio analysis failed: {e}")
        return None

# Replace the method
AdaptiveAudioOrganizer.analyze_audio_content = fixed_analyze_audio_content
# Cell: Fix similar files access
def fixed_find_similar_files(self, filename):
    """Find similar files in learning data - FIXED VERSION"""
    similar = []
    for entry in self.learning_data['classifications']:
        if self.filename_similarity(filename, entry['filename']) > 0.3:
            # Make sure we have the right structure
            classification = entry.get('classification', {})
            similar_entry = {
                'filename': entry['filename'],
                'category': classification.get('category', 'unknown'),
                'mood': classification.get('mood', 'unknown')
            }
            similar.append(similar_entry)
    return similar[:5]

# Replace the method
AdaptiveAudioOrganizer.find_similar_files = fixed_find_similar_files
# Cell: Test with fixes
test_file = resolve_test_file()

print(f"Testing with fixes...")
if test_file and test_file.exists():
    result = organizer.process_file(Path(test_file), dry_run=True)
    if result:
        print("🎉 Test successful!")
        print(f"🏷️ Enhanced filename: {result.get('suggested_filename')}")
        
        # Show BPM detection
        if 'audio_analysis' in result:
            bpm = result['audio_analysis']['raw_features'].get('tempo', 0)
            print(f"🎵 Detected BPM: {bpm:.1f}")
    else:
        print("❌ Classification failed")
else:
    print("❌ Test file not found")
# Cell: Create completely fresh fixed organizer
import os
import shutil
import json
import time
from pathlib import Path
from openai import OpenAI
import mutagen
from collections import defaultdict
import pickle
from datetime import datetime
import librosa
import numpy as np
import hashlib

class FixedAdaptiveAudioOrganizer:
    def __init__(self, openai_api_key, base_directory):
        self.client = OpenAI(api_key=openai_api_key)
        self.base_dir = Path(base_directory)
        
        # Learning system files
        self.learning_data_file = self.base_dir / "04_METADATA_SYSTEM" / "learning_data.pkl"
        self.discovered_categories_file = self.base_dir / "04_METADATA_SYSTEM" / "discovered_categories.json"
        
        # Load existing learning data
        self.learning_data = self.load_learning_data()
        self.discovered_categories = self.load_discovered_categories()
        
        # Base categories that can expand
        self.base_categories = {
            "music_ambient": ["contemplative", "tension_building", "wonder_discovery", "melancholic", "mysterious"],
            "sfx_consciousness": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_human": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_environmental": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_technology": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "voice_element": ["contemplative", "melancholic", "mysterious", "dramatic_punctuation"],
        }
        
        # Dynamic folder mapping
        self.folder_map = self.build_dynamic_folder_map()
        self.audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
        
    def load_learning_data(self):
        """Load historical classification data"""
        if self.learning_data_file.exists():
            with open(self.learning_data_file, 'rb') as f:
                return pickle.load(f)
        return {
            'classifications': [],
            'user_corrections': [],
            'patterns': defaultdict(list),
            'filename_patterns': defaultdict(list)
        }
    
    def save_learning_data(self):
        """Save learning data for future use"""
        self.learning_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.learning_data_file, 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def load_discovered_categories(self):
        """Load dynamically discovered categories"""
        if self.discovered_categories_file.exists():
            with open(self.discovered_categories_file, 'r') as f:
                return json.load(f)
        return {
            'new_moods': [],
            'new_categories': [],
            'new_themes': [],
            'frequency_counts': defaultdict(int)
        }
    
    def save_discovered_categories(self):
        """Save discovered categories"""
        self.discovered_categories_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.discovered_categories_file, 'w') as f:
            json.dump(self.discovered_categories, f, indent=2)
    
    def build_dynamic_folder_map(self):
        """Build folder mapping that includes discovered categories"""
        base_map = {
            "music_ambient + contemplative": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/contemplative/",
            "music_ambient + tension_building": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/tension_building/",
            "music_ambient + wonder_discovery": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/wonder_discovery/",
            "music_ambient + melancholic": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/melancholic/",
            "music_ambient + mysterious": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/mysterious/",
            "sfx_consciousness + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/thought_processing/",
            "sfx_consciousness + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/awakening_emergence/",
            "sfx_consciousness + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/memory_formation/",
            "sfx_human + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/breathing_heartbeat/",
            "sfx_human + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/emotional_responses/",
            "sfx_human + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/environmental_human/",
            "sfx_environmental + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/time_space/",
            "sfx_environmental + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/transformation/",
            "sfx_environmental + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/connection_bridging/",
            "voice_element + contemplative": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/narrator_banks/",
            "voice_element + melancholic": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/processed_vocals/",
            "voice_element + mysterious": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/vocal_textures/",
            "voice_element + dramatic_punctuation": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/character_voices/",
            "default": "01_UNIVERSAL_ASSETS/UNSORTED/"
        }
        return base_map
    
    def get_audio_metadata(self, file_path):
        """Extract audio metadata using mutagen"""
        try:
            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                duration = audio_file.info.length
                return {
                    'duration': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'bitrate': getattr(audio_file.info, 'bitrate', 'Unknown'),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 'Unknown')
                }
        except Exception as e:
            print(f"Could not read metadata for {file_path}: {e}")
        return {'duration': 'Unknown', 'bitrate': 'Unknown', 'sample_rate': 'Unknown'}
    
    def analyze_audio_content(self, file_path):
        """Actually analyze the audio content - FIXED VERSION"""
        try:
            print(f"  🎵 Analyzing audio content...")
            
            # Load first 30 seconds (faster processing)
            y, sr = librosa.load(file_path, duration=30)
            
            features = {}
            
            # Spectral brightness
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate (noisiness vs tonality)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Tempo analysis - FIXED
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy'] = float(np.mean(rms))
            
            # Onset detection (rhythmic activity)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_rate'] = len(onset_frames) / (len(y) / sr)
            
            # Spectral contrast (timbral texture)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            
            # Interpret features into human-readable descriptions
            interpretation = self.interpret_audio_features(features)
            
            return {
                'raw_features': features,
                'interpretation': interpretation
            }
            
        except Exception as e:
            print(f"  ⚠️  Audio analysis failed: {e}")
            return None

    def interpret_audio_features(self, features):
        """Convert raw features to human-readable descriptions"""
        interpretation = {}
        
        # Brightness (spectral centroid)
        if features['spectral_centroid_mean'] > 3000:
            interpretation['brightness'] = 'bright'
        elif features['spectral_centroid_mean'] > 1000:
            interpretation['brightness'] = 'balanced'
        else:
            interpretation['brightness'] = 'dark'
        
        # Texture (zero crossing rate)
        if features['zero_crossing_rate'] > 0.1:
            interpretation['texture'] = 'noisy'
        elif features['zero_crossing_rate'] > 0.05:
            interpretation['texture'] = 'moderate'
        else:
            interpretation['texture'] = 'tonal'
        
        # Energy level
        if features['rms_energy'] > 0.1:
            interpretation['energy'] = 'high'
        elif features['rms_energy'] > 0.05:
            interpretation['energy'] = 'medium'
        else:
            interpretation['energy'] = 'low'
        
        # Tempo interpretation
        if features['tempo'] > 120:
            interpretation['pace'] = 'fast'
        elif features['tempo'] > 80:
            interpretation['pace'] = 'moderate'
        else:
            interpretation['pace'] = 'slow'
        
        # Rhythmic activity
        if features['onset_rate'] > 2:
            interpretation['rhythmic_activity'] = 'high'
        elif features['onset_rate'] > 0.5:
            interpretation['rhythmic_activity'] = 'moderate'
        else:
            interpretation['rhythmic_activity'] = 'low'
        
        return interpretation

    def analyze_filename_patterns(self, filename):
        """Extract patterns from filename that might indicate content type"""
        patterns = []
        filename_lower = filename.lower()
        
        # Common audio descriptors
        descriptors = {
            'ambient': ['ambient', 'atmosphere', 'atmos', 'background', 'bg'],
            'percussion': ['drum', 'beat', 'perc', 'rhythm', 'kick', 'snare'],
            'vocal': ['vocal', 'voice', 'speech', 'talk', 'narr', 'dialogue'],
            'nature': ['nature', 'wind', 'rain', 'water', 'forest', 'bird'],
            'mechanical': ['mech', 'robot', 'machine', 'tech', 'digital', 'synth'],
            'emotional': ['sad', 'happy', 'dark', 'bright', 'calm', 'tense'],
            'temporal': ['intro', 'outro', 'loop', 'oneshot', 'sustained']
        }
        
        for category, keywords in descriptors.items():
            if any(keyword in filename_lower for keyword in keywords):
                patterns.append(category)
        
        return patterns
    
    def find_similar_files(self, filename):
        """Find similar files in learning data - FIXED VERSION"""
        similar = []
        for entry in self.learning_data['classifications']:
            if self.filename_similarity(filename, entry['filename']) > 0.3:
                # Make sure we have the right structure - FIXED
                classification = entry.get('classification', {})
                similar_entry = {
                    'filename': entry['filename'],
                    'category': classification.get('category', 'unknown'),
                    'mood': classification.get('mood', 'unknown')
                }
                similar.append(similar_entry)
        return similar[:5]
    
    def filename_similarity(self, filename1, filename2):
        """Calculate similarity between filenames"""
        words1 = set(filename1.lower().replace('_', ' ').split())
        words2 = set(filename2.lower().replace('_', ' ').split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def generate_enhanced_filename(self, file_path, classification, audio_analysis):
        """Generate descriptive filename including BPM and other useful info"""
        
        # Extract key info
        category = classification.get('category', 'unknown')
        mood = classification.get('mood', 'unknown')
        intensity = classification.get('intensity', 'unknown')
        energy = classification.get('energy_level', 0)
        
        # Get audio characteristics
        bpm = None
        brightness = None
        texture = None
        
        if audio_analysis:
            raw_features = audio_analysis.get('raw_features', {})
            interpretation = audio_analysis.get('interpretation', {})
            
            bpm = raw_features.get('tempo', 0)
            brightness = interpretation.get('brightness', '')
            texture = interpretation.get('texture', '')
        
        # Build filename components
        components = []
        
        # 1. Category prefix
        if category.startswith('music_'):
            components.append('MUS')
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '').upper()
            components.append(f'SFX_{sfx_type[:4]}')
        elif category.startswith('voice_'):
            components.append('VOX')
        else:
            components.append(category.upper()[:6])
        
        # 2. BPM (if detected and makes sense)
        if bpm and bpm > 30 and bpm < 300:
            components.append(f"{int(bpm)}bpm")
        
        # 3. Mood/Energy
        mood_short = mood[:4].upper() if mood else ""
        if mood_short:
            components.append(mood_short)
        
        # 4. Energy level
        if energy:
            components.append(f"E{energy}")
        
        # 5. Audio characteristics
        if brightness in ['bright', 'dark']:
            components.append(brightness[:3].upper())
        
        if texture == 'tonal':
            components.append('TON')
        elif texture == 'noisy':
            components.append('NOI')
        
        # 6. Intensity indicator
        intensity_map = {
            'subtle_background': 'BG',
            'narrative_support': 'SUP', 
            'dramatic_punctuation': 'DRA'
        }
        if intensity in intensity_map:
            components.append(intensity_map[intensity])
        
        # Join components
        new_filename = '_'.join(components)
        
        # Keep original extension
        original_ext = file_path.suffix
        
        # Add uniqueness
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:4]
        
        final_filename = f"{new_filename}_{file_hash}{original_ext}"
        
        return final_filename
    
    def classify_audio_file(self, file_path, user_description=""):
        """Classify using BOTH filename patterns AND actual audio analysis"""
        
        metadata = self.get_audio_metadata(file_path)
        audio_analysis = self.analyze_audio_content(file_path)
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        similar_files = self.find_similar_files(file_path.name)  # Now fixed
        
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Build comprehensive context
        context = ""
        
        # Only add similar files if we found any
        if similar_files:
            context += f"\nSIMILAR FILES:\n"
            for similar in similar_files[:3]:
                context += f"- {similar['filename']}: {similar['category']} + {similar['mood']}\n"
        
        if filename_patterns:
            context += f"\nFILENAME PATTERNS: {', '.join(filename_patterns)}\n"
        
        # Add audio analysis insights
        if audio_analysis:
            interp = audio_analysis['interpretation']
            raw = audio_analysis['raw_features']
            context += f"\nAUDIO ANALYSIS:\n"
            context += f"- Brightness: {interp.get('brightness', 'unknown')}\n"
            context += f"- Texture: {interp.get('texture', 'unknown')}\n"
            context += f"- Energy: {interp.get('energy', 'unknown')}\n"
            context += f"- Pace: {interp.get('pace', 'unknown')}\n"
            context += f"- Rhythmic Activity: {interp.get('rhythmic_activity', 'unknown')}\n"
            context += f"- Tempo: {raw.get('tempo', 0):.1f} BPM\n"
        
        # Get dynamic categories
        all_categories = list(self.base_categories.keys()) + self.discovered_categories.get('new_categories', [])
        all_moods = []
        for cat_moods in self.base_categories.values():
            all_moods.extend(cat_moods)
        all_moods.extend(self.discovered_categories.get('new_moods', []))
        all_moods = list(set(all_moods))
        
        prompt = f"""You are an expert audio librarian with access to both filename analysis AND actual audio content analysis. Use this combined information for the most accurate classification.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
{context}

CLASSIFICATION FRAMEWORK:
Available categories: {', '.join(all_categories)}
Available moods: {', '.join(all_moods)}

INSTRUCTIONS:
- Combine filename clues with actual audio characteristics
- Audio analysis should take precedence over filename when they conflict
- Suggest new categories/moods if the audio doesn't fit existing ones
- Be precise about BPM if detected - this will be used in the filename

Return JSON:
{{
  "category": "category_name",
  "mood": "mood_name", 
  "intensity": "intensity_level",
  "energy_level": 1-10,
  "tags": ["tag1", "tag2"],
  "thematic_notes": "How this enhances AI consciousness storytelling",
  "confidence": 0.0-1.0,
  "reasoning": "Why you chose this, combining filename and audio insights",
  "discovered_elements": ["new patterns found"],
  "audio_insights": "What the audio analysis revealed"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(raw_response)
            
            # Generate enhanced filename with BPM
            enhanced_filename = self.generate_enhanced_filename(file_path, classification, audio_analysis)
            classification['suggested_filename'] = enhanced_filename
            
            # Store audio analysis
            if audio_analysis:
                classification['audio_analysis'] = audio_analysis
            
            self.learn_from_classification(file_path, classification)
            return classification
            
        except Exception as e:
            print(f"Enhanced classification failed: {e}")
            return None
    
    def learn_from_classification(self, file_path, classification):
        """Learn from each classification to improve future ones"""
        
        learning_entry = {
            'filename': file_path.name,
            'classification': classification,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        
        self.learning_data['classifications'].append(learning_entry)
        
        # Track new categories/moods
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        
        # Check for new categories
        if category not in [cat for cats in self.base_categories.keys() for cat in cats]:
            if category not in self.discovered_categories['new_categories']:
                self.discovered_categories['new_categories'].append(category)
                print(f"🆕 Discovered new category: {category}")
        
        # Check for new moods
        all_base_moods = [mood for moods in self.base_categories.values() for mood in moods]
        if mood not in all_base_moods:
            if mood not in self.discovered_categories['new_moods']:
                self.discovered_categories['new_moods'].append(mood)
                print(f"🆕 Discovered new mood: {mood}")
        
        # Update frequency counts
        self.discovered_categories['frequency_counts'][f"{category}+{mood}"] += 1
        
        # Save learning data
        self.save_learning_data()
        self.save_discovered_categories()
        
        # Rebuild folder map with new discoveries
        self.folder_map = self.build_dynamic_folder_map()
    
    def determine_target_folder(self, classification):
        """Determine target folder, creating new ones if needed"""
        if not classification:
            return self.folder_map["default"]
        
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        intensity = classification.get('intensity', '')
        
        # Try exact match first
        key = f"{category} + {mood}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Try with intensity
        key = f"{category} + {intensity}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Create new folder path for discovered categories
        if category.startswith('music_'):
            new_path = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '')
            new_path = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
        elif category.startswith('voice_'):
            new_path = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
        else:
            new_path = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        # Add to folder map
        self.folder_map[key] = new_path
        print(f"🆕 Created new folder mapping: {key} → {new_path}")
        
        return new_path
    
    def process_file(self, file_path, user_description="", dry_run=True):
        """Process file with adaptive learning"""
        print(f"\n🔍 Processing: {file_path.name}")
        
        classification = self.classify_audio_file(file_path, user_description)
        
        if classification:
            target_folder = self.determine_target_folder(classification)
            
            print(f"  📂 Category: {classification.get('category', 'unknown')}")
            print(f"  🎭 Mood: {classification.get('mood', 'unknown')}")
            print(f"  ⚡ Intensity: {classification.get('intensity', 'unknown')}")
            print(f"  🔥 Energy: {classification.get('energy_level', 0)}/10")
            print(f"  🎯 Confidence: {classification.get('confidence', 0):.1%}")
            print(f"  📍 Target: {target_folder}")
            print(f"  🏷️ Enhanced filename: {classification.get('suggested_filename', 'None')}")
            print(f"  🏷️ Tags: {', '.join(classification.get('tags', []))}")
            
            if classification.get('discovered_elements'):
                print(f"  🆕 Discovered: {', '.join(classification.get('discovered_elements', []))}")
            
            if not dry_run:
                print(f"  ✅ Would move to: {target_folder}")
            else:
                print(f"  🔄 [DRY RUN] Would move to: {target_folder}")
            
            return classification
        else:
            print(f"  ❌ Classification failed")
            return None
    
    def show_learning_stats(self):
        """Display learning statistics"""
        print(f"\n📊 LEARNING STATISTICS")
        print(f"=" * 50)
        
        total_files = len(self.learning_data['classifications'])
        print(f"Total files processed: {total_files}")
        
        print(f"\n🆕 DISCOVERED CATEGORIES:")
        for category in self.discovered_categories.get('new_categories', []):
            print(f"  - {category}")
        
        print(f"\n🆕 DISCOVERED MOODS:")
        for mood in self.discovered_categories.get('new_moods', []):
            print(f"  - {mood}")
        
        print(f"\n📈 MOST COMMON COMBINATIONS:")
        freq_counts = self.discovered_categories.get('frequency_counts', {})
        for combo, count in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {combo}: {count} files")

print("🔧 Fixed Enhanced AdaptiveAudioOrganizer created!")# Cell: Create completely fresh fixed organizer
import os
import shutil
import json
import time
from pathlib import Path
from openai import OpenAI
import mutagen
from collections import defaultdict
import pickle
from datetime import datetime
import librosa
import numpy as np
import hashlib

class FixedAdaptiveAudioOrganizer:
    def __init__(self, openai_api_key, base_directory):
        self.client = OpenAI(api_key=openai_api_key)
        self.base_dir = Path(base_directory)
        
        # Learning system files
        self.learning_data_file = self.base_dir / "04_METADATA_SYSTEM" / "learning_data.pkl"
        self.discovered_categories_file = self.base_dir / "04_METADATA_SYSTEM" / "discovered_categories.json"
        
        # Load existing learning data
        self.learning_data = self.load_learning_data()
        self.discovered_categories = self.load_discovered_categories()
        
        # Base categories that can expand
        self.base_categories = {
            "music_ambient": ["contemplative", "tension_building", "wonder_discovery", "melancholic", "mysterious"],
            "sfx_consciousness": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_human": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_environmental": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "sfx_technology": ["subtle_background", "narrative_support", "dramatic_punctuation"],
            "voice_element": ["contemplative", "melancholic", "mysterious", "dramatic_punctuation"],
        }
        
        # Dynamic folder mapping
        self.folder_map = self.build_dynamic_folder_map()
        self.audio_extensions = {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}
        
    def load_learning_data(self):
        """Load historical classification data"""
        if self.learning_data_file.exists():
            with open(self.learning_data_file, 'rb') as f:
                return pickle.load(f)
        return {
            'classifications': [],
            'user_corrections': [],
            'patterns': defaultdict(list),
            'filename_patterns': defaultdict(list)
        }
    
    def save_learning_data(self):
        """Save learning data for future use"""
        self.learning_data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.learning_data_file, 'wb') as f:
            pickle.dump(self.learning_data, f)
    
    def load_discovered_categories(self):
        """Load dynamically discovered categories"""
        if self.discovered_categories_file.exists():
            with open(self.discovered_categories_file, 'r') as f:
                return json.load(f)
        return {
            'new_moods': [],
            'new_categories': [],
            'new_themes': [],
            'frequency_counts': defaultdict(int)
        }
    
    def save_discovered_categories(self):
        """Save discovered categories"""
        self.discovered_categories_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.discovered_categories_file, 'w') as f:
            json.dump(self.discovered_categories, f, indent=2)
    
    def build_dynamic_folder_map(self):
        """Build folder mapping that includes discovered categories"""
        base_map = {
            "music_ambient + contemplative": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/contemplative/",
            "music_ambient + tension_building": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/tension_building/",
            "music_ambient + wonder_discovery": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/wonder_discovery/",
            "music_ambient + melancholic": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/melancholic/",
            "music_ambient + mysterious": "01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/mysterious/",
            "sfx_consciousness + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/thought_processing/",
            "sfx_consciousness + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/awakening_emergence/",
            "sfx_consciousness + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/consciousness/memory_formation/",
            "sfx_human + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/breathing_heartbeat/",
            "sfx_human + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/emotional_responses/",
            "sfx_human + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/human_elements/environmental_human/",
            "sfx_environmental + subtle_background": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/time_space/",
            "sfx_environmental + narrative_support": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/transformation/",
            "sfx_environmental + dramatic_punctuation": "01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/abstract_conceptual/connection_bridging/",
            "voice_element + contemplative": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/narrator_banks/",
            "voice_element + melancholic": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/processed_vocals/",
            "voice_element + mysterious": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/vocal_textures/",
            "voice_element + dramatic_punctuation": "01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/character_voices/",
            "default": "01_UNIVERSAL_ASSETS/UNSORTED/"
        }
        return base_map
    
    def get_audio_metadata(self, file_path):
        """Extract audio metadata using mutagen"""
        try:
            audio_file = mutagen.File(file_path)
            if audio_file is not None:
                duration = audio_file.info.length
                return {
                    'duration': f"{int(duration // 60)}:{int(duration % 60):02d}",
                    'bitrate': getattr(audio_file.info, 'bitrate', 'Unknown'),
                    'sample_rate': getattr(audio_file.info, 'sample_rate', 'Unknown')
                }
        except Exception as e:
            print(f"Could not read metadata for {file_path}: {e}")
        return {'duration': 'Unknown', 'bitrate': 'Unknown', 'sample_rate': 'Unknown'}
    
    def analyze_audio_content(self, file_path):
        """Actually analyze the audio content - FIXED VERSION"""
        try:
            print(f"  🎵 Analyzing audio content...")
            
            # Load first 30 seconds (faster processing)
            y, sr = librosa.load(file_path, duration=30)
            
            features = {}
            
            # Spectral brightness
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # Spectral rolloff (high frequency content)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate (noisiness vs tonality)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Tempo analysis - FIXED
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
            
            # Energy (RMS)
            rms = librosa.feature.rms(y=y)[0]
            features['rms_energy'] = float(np.mean(rms))
            
            # Onset detection (rhythmic activity)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            features['onset_rate'] = len(onset_frames) / (len(y) / sr)
            
            # Spectral contrast (timbral texture)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['spectral_contrast_mean'] = float(np.mean(contrast))
            
            # Interpret features into human-readable descriptions
            interpretation = self.interpret_audio_features(features)
            
            return {
                'raw_features': features,
                'interpretation': interpretation
            }
            
        except Exception as e:
            print(f"  ⚠️  Audio analysis failed: {e}")
            return None

    def interpret_audio_features(self, features):
        """Convert raw features to human-readable descriptions"""
        interpretation = {}
        
        # Brightness (spectral centroid)
        if features['spectral_centroid_mean'] > 3000:
            interpretation['brightness'] = 'bright'
        elif features['spectral_centroid_mean'] > 1000:
            interpretation['brightness'] = 'balanced'
        else:
            interpretation['brightness'] = 'dark'
        
        # Texture (zero crossing rate)
        if features['zero_crossing_rate'] > 0.1:
            interpretation['texture'] = 'noisy'
        elif features['zero_crossing_rate'] > 0.05:
            interpretation['texture'] = 'moderate'
        else:
            interpretation['texture'] = 'tonal'
        
        # Energy level
        if features['rms_energy'] > 0.1:
            interpretation['energy'] = 'high'
        elif features['rms_energy'] > 0.05:
            interpretation['energy'] = 'medium'
        else:
            interpretation['energy'] = 'low'
        
        # Tempo interpretation
        if features['tempo'] > 120:
            interpretation['pace'] = 'fast'
        elif features['tempo'] > 80:
            interpretation['pace'] = 'moderate'
        else:
            interpretation['pace'] = 'slow'
        
        # Rhythmic activity
        if features['onset_rate'] > 2:
            interpretation['rhythmic_activity'] = 'high'
        elif features['onset_rate'] > 0.5:
            interpretation['rhythmic_activity'] = 'moderate'
        else:
            interpretation['rhythmic_activity'] = 'low'
        
        return interpretation

    def analyze_filename_patterns(self, filename):
        """Extract patterns from filename that might indicate content type"""
        patterns = []
        filename_lower = filename.lower()
        
        # Common audio descriptors
        descriptors = {
            'ambient': ['ambient', 'atmosphere', 'atmos', 'background', 'bg'],
            'percussion': ['drum', 'beat', 'perc', 'rhythm', 'kick', 'snare'],
            'vocal': ['vocal', 'voice', 'speech', 'talk', 'narr', 'dialogue'],
            'nature': ['nature', 'wind', 'rain', 'water', 'forest', 'bird'],
            'mechanical': ['mech', 'robot', 'machine', 'tech', 'digital', 'synth'],
            'emotional': ['sad', 'happy', 'dark', 'bright', 'calm', 'tense'],
            'temporal': ['intro', 'outro', 'loop', 'oneshot', 'sustained']
        }
        
        for category, keywords in descriptors.items():
            if any(keyword in filename_lower for keyword in keywords):
                patterns.append(category)
        
        return patterns
    
    def find_similar_files(self, filename):
        """Find similar files in learning data - FIXED VERSION"""
        similar = []
        for entry in self.learning_data['classifications']:
            if self.filename_similarity(filename, entry['filename']) > 0.3:
                # Make sure we have the right structure - FIXED
                classification = entry.get('classification', {})
                similar_entry = {
                    'filename': entry['filename'],
                    'category': classification.get('category', 'unknown'),
                    'mood': classification.get('mood', 'unknown')
                }
                similar.append(similar_entry)
        return similar[:5]
    
    def filename_similarity(self, filename1, filename2):
        """Calculate similarity between filenames"""
        words1 = set(filename1.lower().replace('_', ' ').split())
        words2 = set(filename2.lower().replace('_', ' ').split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def generate_enhanced_filename(self, file_path, classification, audio_analysis):
        """Generate descriptive filename including BPM and other useful info"""
        
        # Extract key info
        category = classification.get('category', 'unknown')
        mood = classification.get('mood', 'unknown')
        intensity = classification.get('intensity', 'unknown')
        energy = classification.get('energy_level', 0)
        
        # Get audio characteristics
        bpm = None
        brightness = None
        texture = None
        
        if audio_analysis:
            raw_features = audio_analysis.get('raw_features', {})
            interpretation = audio_analysis.get('interpretation', {})
            
            bpm = raw_features.get('tempo', 0)
            brightness = interpretation.get('brightness', '')
            texture = interpretation.get('texture', '')
        
        # Build filename components
        components = []
        
        # 1. Category prefix
        if category.startswith('music_'):
            components.append('MUS')
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '').upper()
            components.append(f'SFX_{sfx_type[:4]}')
        elif category.startswith('voice_'):
            components.append('VOX')
        else:
            components.append(category.upper()[:6])
        
        # 2. BPM (if detected and makes sense)
        if bpm and bpm > 30 and bpm < 300:
            components.append(f"{int(bpm)}bpm")
        
        # 3. Mood/Energy
        mood_short = mood[:4].upper() if mood else ""
        if mood_short:
            components.append(mood_short)
        
        # 4. Energy level
        if energy:
            components.append(f"E{energy}")
        
        # 5. Audio characteristics
        if brightness in ['bright', 'dark']:
            components.append(brightness[:3].upper())
        
        if texture == 'tonal':
            components.append('TON')
        elif texture == 'noisy':
            components.append('NOI')
        
        # 6. Intensity indicator
        intensity_map = {
            'subtle_background': 'BG',
            'narrative_support': 'SUP', 
            'dramatic_punctuation': 'DRA'
        }
        if intensity in intensity_map:
            components.append(intensity_map[intensity])
        
        # Join components
        new_filename = '_'.join(components)
        
        # Keep original extension
        original_ext = file_path.suffix
        
        # Add uniqueness
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:4]
        
        final_filename = f"{new_filename}_{file_hash}{original_ext}"
        
        return final_filename
    
    def classify_audio_file(self, file_path, user_description=""):
        """Classify using BOTH filename patterns AND actual audio analysis"""
        
        metadata = self.get_audio_metadata(file_path)
        audio_analysis = self.analyze_audio_content(file_path)
        filename_patterns = self.analyze_filename_patterns(file_path.name)
        similar_files = self.find_similar_files(file_path.name)  # Now fixed
        
        file_stats = os.stat(file_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        
        # Build comprehensive context
        context = ""
        
        # Only add similar files if we found any
        if similar_files:
            context += f"\nSIMILAR FILES:\n"
            for similar in similar_files[:3]:
                context += f"- {similar['filename']}: {similar['category']} + {similar['mood']}\n"
        
        if filename_patterns:
            context += f"\nFILENAME PATTERNS: {', '.join(filename_patterns)}\n"
        
        # Add audio analysis insights
        if audio_analysis:
            interp = audio_analysis['interpretation']
            raw = audio_analysis['raw_features']
            context += f"\nAUDIO ANALYSIS:\n"
            context += f"- Brightness: {interp.get('brightness', 'unknown')}\n"
            context += f"- Texture: {interp.get('texture', 'unknown')}\n"
            context += f"- Energy: {interp.get('energy', 'unknown')}\n"
            context += f"- Pace: {interp.get('pace', 'unknown')}\n"
            context += f"- Rhythmic Activity: {interp.get('rhythmic_activity', 'unknown')}\n"
            context += f"- Tempo: {raw.get('tempo', 0):.1f} BPM\n"
        
        # Get dynamic categories
        all_categories = list(self.base_categories.keys()) + self.discovered_categories.get('new_categories', [])
        all_moods = []
        for cat_moods in self.base_categories.values():
            all_moods.extend(cat_moods)
        all_moods.extend(self.discovered_categories.get('new_moods', []))
        all_moods = list(set(all_moods))
        
        prompt = f"""You are an expert audio librarian with access to both filename analysis AND actual audio content analysis. Use this combined information for the most accurate classification.

AUDIO DETAILS:
- Filename: {file_path.name}
- Duration: {metadata['duration']}
- File size: {file_size_mb:.2f} MB
- File type: {file_path.suffix}
{context}

CLASSIFICATION FRAMEWORK:
Available categories: {', '.join(all_categories)}
Available moods: {', '.join(all_moods)}

INSTRUCTIONS:
- Combine filename clues with actual audio characteristics
- Audio analysis should take precedence over filename when they conflict
- Suggest new categories/moods if the audio doesn't fit existing ones
- Be precise about BPM if detected - this will be used in the filename

Return JSON:
{{
  "category": "category_name",
  "mood": "mood_name", 
  "intensity": "intensity_level",
  "energy_level": 1-10,
  "tags": ["tag1", "tag2"],
  "thematic_notes": "How this enhances AI consciousness storytelling",
  "confidence": 0.0-1.0,
  "reasoning": "Why you chose this, combining filename and audio insights",
  "discovered_elements": ["new patterns found"],
  "audio_insights": "What the audio analysis revealed"
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            if raw_response.startswith('```json'):
                raw_response = raw_response.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(raw_response)
            
            # Generate enhanced filename with BPM
            enhanced_filename = self.generate_enhanced_filename(file_path, classification, audio_analysis)
            classification['suggested_filename'] = enhanced_filename
            
            # Store audio analysis
            if audio_analysis:
                classification['audio_analysis'] = audio_analysis
            
            self.learn_from_classification(file_path, classification)
            return classification
            
        except Exception as e:
            print(f"Enhanced classification failed: {e}")
            return None
    
    def learn_from_classification(self, file_path, classification):
        """Learn from each classification to improve future ones"""
        
        learning_entry = {
            'filename': file_path.name,
            'classification': classification,
            'timestamp': datetime.now().isoformat(),
            'file_path': str(file_path)
        }
        
        self.learning_data['classifications'].append(learning_entry)
        
        # Track new categories/moods
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        
        # Check for new categories
        if category not in [cat for cats in self.base_categories.keys() for cat in cats]:
            if category not in self.discovered_categories['new_categories']:
                self.discovered_categories['new_categories'].append(category)
                print(f"🆕 Discovered new category: {category}")
        
        # Check for new moods
        all_base_moods = [mood for moods in self.base_categories.values() for mood in moods]
        if mood not in all_base_moods:
            if mood not in self.discovered_categories['new_moods']:
                self.discovered_categories['new_moods'].append(mood)
                print(f"🆕 Discovered new mood: {mood}")
        
        # Update frequency counts
        self.discovered_categories['frequency_counts'][f"{category}+{mood}"] += 1
        
        # Save learning data
        self.save_learning_data()
        self.save_discovered_categories()
        
        # Rebuild folder map with new discoveries
        self.folder_map = self.build_dynamic_folder_map()
    
    def determine_target_folder(self, classification):
        """Determine target folder, creating new ones if needed"""
        if not classification:
            return self.folder_map["default"]
        
        category = classification.get('category', '')
        mood = classification.get('mood', '')
        intensity = classification.get('intensity', '')
        
        # Try exact match first
        key = f"{category} + {mood}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Try with intensity
        key = f"{category} + {intensity}"
        if key in self.folder_map:
            return self.folder_map[key]
        
        # Create new folder path for discovered categories
        if category.startswith('music_'):
            new_path = f"01_UNIVERSAL_ASSETS/MUSIC_LIBRARY/by_mood/{mood}/"
        elif category.startswith('sfx_'):
            sfx_type = category.replace('sfx_', '')
            new_path = f"01_UNIVERSAL_ASSETS/SFX_LIBRARY/by_category/{sfx_type}/{mood}/"
        elif category.startswith('voice_'):
            new_path = f"01_UNIVERSAL_ASSETS/VOICE_ELEMENTS/{mood}/"
        else:
            new_path = f"01_UNIVERSAL_ASSETS/EXPERIMENTAL/{category}/{mood}/"
        
        # Add to folder map
        self.folder_map[key] = new_path
        print(f"🆕 Created new folder mapping: {key} → {new_path}")
        
        return new_path
    
    def process_file(self, file_path, user_description="", dry_run=True):
        """Process file with adaptive learning"""
        print(f"\n🔍 Processing: {file_path.name}")
        
        classification = self.classify_audio_file(file_path, user_description)
        
        if classification:
            target_folder = self.determine_target_folder(classification)
            
            print(f"  📂 Category: {classification.get('category', 'unknown')}")
            print(f"  🎭 Mood: {classification.get('mood', 'unknown')}")
            print(f"  ⚡ Intensity: {classification.get('intensity', 'unknown')}")
            print(f"  🔥 Energy: {classification.get('energy_level', 0)}/10")
            print(f"  🎯 Confidence: {classification.get('confidence', 0):.1%}")
            print(f"  📍 Target: {target_folder}")
            print(f"  🏷️ Enhanced filename: {classification.get('suggested_filename', 'None')}")
            print(f"  🏷️ Tags: {', '.join(classification.get('tags', []))}")
            
            if classification.get('discovered_elements'):
                print(f"  🆕 Discovered: {', '.join(classification.get('discovered_elements', []))}")
            
            if not dry_run:
                print(f"  ✅ Would move to: {target_folder}")
            else:
                print(f"  🔄 [DRY RUN] Would move to: {target_folder}")
            
            return classification
        else:
            print(f"  ❌ Classification failed")
            return None
    
    def show_learning_stats(self):
        """Display learning statistics"""
        print(f"\n📊 LEARNING STATISTICS")
        print(f"=" * 50)
        
        total_files = len(self.learning_data['classifications'])
        print(f"Total files processed: {total_files}")
        
        print(f"\n🆕 DISCOVERED CATEGORIES:")
        for category in self.discovered_categories.get('new_categories', []):
            print(f"  - {category}")
        
        print(f"\n🆕 DISCOVERED MOODS:")
        for mood in self.discovered_categories.get('new_moods', []):
            print(f"  - {mood}")
        
        print(f"\n📈 MOST COMMON COMBINATIONS:")
        freq_counts = self.discovered_categories.get('frequency_counts', {})
        for combo, count in sorted(freq_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {combo}: {count} files")

print("🔧 Fixed Enhanced AdaptiveAudioOrganizer created!")
# Cell: Create fixed organizer
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("❌ Please set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY='your-api-key-here'")
    exit(1)
BASE_DIRECTORY = os.getenv('AUDIOAI_BASE_DIRECTORY')
if not BASE_DIRECTORY:
    print("❌ Please set AUDIOAI_BASE_DIRECTORY environment variable")
    print("   export AUDIOAI_BASE_DIRECTORY='/path/to/your/audio/library'")
    exit(1)
organizer = FixedAdaptiveAudioOrganizer(OPENAI_API_KEY, BASE_DIRECTORY)

print("✅ Fixed organizer created!")
# Cell: Test fixed organizer
test_file = resolve_test_file()

if Path(test_file).exists():
    result = organizer.process_file(Path(test_file), dry_run=True)
    if result:
        print("🎉 SUCCESS! Fixed organizer working!")
        print(f"🏷️ Enhanced filename: {result.get('suggested_filename')}")
        if 'audio_analysis' in result:
            bpm = result['audio_analysis']['raw_features'].get('tempo', 0)
            print(f"🎵 Detected BPM: {bpm:.1f}")
else:
    print("❌ Test file not found")
# Cell: Full System Sweep with Enhanced Audio Analysis
print("🚀 LAUNCHING FULL SYSTEM SWEEP!")
print("=" * 70)
print("🎵 Audio analysis: ACTIVE")
print("🏷️ BPM detection: ACTIVE") 
print("🧠 Adaptive learning: ACTIVE")
print("🆕 Category discovery: ACTIVE")
print("=" * 70)

# Find ALL audio files recursively
print("🔍 Scanning your entire audio library...")
all_audio_files = []

#scan_path = BASE_DIRECTORY
scan_path = BASE_DIRECTORY

for file_path in scan_path.rglob('*'):
    if file_path.is_file() and file_path.suffix.lower() in {'.mp3', '.wav', '.aiff', '.m4a', '.flac', '.ogg', '.wma'}:
        # Skip already organized files to avoid re-processing
        if not any(skip_folder in str(file_path) for skip_folder in ['01_UNIVERSAL_ASSETS', 'THEMATIC_COLLECTIONS']):
            all_audio_files.append(file_path)

print(f"📊 Found {len(all_audio_files)} audio files to analyze!")

# Show what folders we found
folders_found = {}
for file_path in all_audio_files:
    folder = str(file_path.parent.relative_to(scan_path))
    folders_found[folder] = folders_found.get(folder, 0) + 1

print(f"\n📂 Files by folder:")
for folder, count in sorted(folders_found.items()):
    print(f"  📁 {folder}: {count} files")

print(f"\n⏱️  Estimated processing time: {len(all_audio_files) * 0.5:.1f} minutes")
print(f"💡 This will analyze audio content, detect BPMs, and learn patterns")
# Cell: Execute the Full Enhanced Sweep
import time
from datetime import datetime

print("🎵 STARTING ENHANCED AUDIO ANALYSIS SWEEP!")
print("⚠️  This will take time due to actual audio analysis...")
print("💡 You can stop anytime with Ctrl+C")
print("-" * 70)

# Track progress
processed_count = 0
failed_count = 0
start_time = time.time()
discovered_bpms = []
discovered_categories = set()
discovered_moods = set()

for i, file_path in enumerate(all_audio_files):
    try:
        # Progress tracking
        elapsed = time.time() - start_time
        if i > 0:
            avg_time_per_file = elapsed / i
            remaining_files = len(all_audio_files) - i
            eta_seconds = remaining_files * avg_time_per_file
            eta_minutes = eta_seconds / 60
            
            print(f"\n[{i+1}/{len(all_audio_files)}] ⏱️  ETA: {eta_minutes:.1f} min")
        else:
            print(f"\n[{i+1}/{len(all_audio_files)}]")
        
        # Process the file
        result = organizer.process_file(file_path, dry_run=True)
        
        if result:
            processed_count += 1
            
            # Track discoveries
            if 'audio_analysis' in result:
                bpm = result['audio_analysis']['raw_features'].get('tempo', 0)
                if 30 < bpm < 300:  # Valid BPM range
                    discovered_bpms.append(bpm)
            
            category = result.get('category', '')
            mood = result.get('mood', '')
            if category:
                discovered_categories.add(category)
            if mood:
                discovered_moods.add(mood)
            
            # Show interesting discoveries
            if result.get('discovered_elements'):
                discoveries = result.get('discovered_elements', [])
                print(f"  🆕 NEW: {', '.join(discoveries)}")
            
            # Show BPM if detected
            if 'audio_analysis' in result:
                bpm = result['audio_analysis']['raw_features'].get('tempo', 0)
                if 30 < bpm < 300:
                    print(f"  🎵 BPM: {bpm:.0f}")
            
        else:
            failed_count += 1
            print(f"  ❌ Classification failed")
        
        # Show learning progress every 25 files
        if (i + 1) % 25 == 0:
            print(f"\n📈 PROGRESS UPDATE after {i+1} files:")
            print(f"  ✅ Processed: {processed_count}")
            print(f"  ❌ Failed: {failed_count}")
            print(f"  🎵 BPMs detected: {len(discovered_bpms)}")
            print(f"  📂 Categories: {len(discovered_categories)}")
            print(f"  🎭 Moods: {len(discovered_moods)}")
            
            if discovered_bpms:
                avg_bpm = sum(discovered_bpms) / len(discovered_bpms)
                print(f"  📊 Average BPM: {avg_bpm:.0f}")
            
            organizer.show_learning_stats()
            print("-" * 50)
        
        # Don't overwhelm the API
        time.sleep(0.3)
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  SWEEP INTERRUPTED by user after {i+1} files")
        break
    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        failed_count += 1

# Final results
total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"🎉 SWEEP COMPLETE!")
print(f"⏱️  Total time: {total_time/60:.1f} minutes")
print(f"📊 FINAL RESULTS:")
print(f"  ✅ Successfully processed: {processed_count}")
print(f"  ❌ Failed: {failed_count}")
print(f"  📈 Total files: {len(all_audio_files)}")
print(f"  🎵 BPMs detected: {len(discovered_bpms)}")

if discovered_bpms:
    print(f"  📊 BPM range: {min(discovered_bpms):.0f} - {max(discovered_bpms):.0f}")
    print(f"  📊 Average BPM: {sum(discovered_bpms)/len(discovered_bpms):.0f}")

print(f"  📂 Categories discovered: {len(discovered_categories)}")
print(f"  🎭 Moods discovered: {len(discovered_moods)}")

# Show final learning stats
print(f"\n📊 FINAL LEARNING STATISTICS:")
organizer.show_learning_stats()

print(f"\n🎯 Your audio library has been analyzed!")
print(f"💾 Learning data saved to: {organizer.learning_data_file}")
print(f"🔍 Next: Review results and run with dry_run=False to actually organize files")
# Cell: Quick fixes for the issues
def fixed_learn_from_classification(self, file_path, classification):
    """Learn from each classification - FIXED VERSION"""
    
    learning_entry = {
        'filename': file_path.name,
        'classification': classification,
        'timestamp': datetime.now().isoformat(),
        'file_path': str(file_path)
    }
    
    self.learning_data['classifications'].append(learning_entry)
    
    # Track new categories/moods - FIXED
    category = classification.get('category', '')
    mood = classification.get('mood', '')
    
    # Check for new categories - FIXED logic
    if category and category not in self.base_categories:
        if category not in self.discovered_categories['new_categories']:
            self.discovered_categories['new_categories'].append(category)
            print(f"🆕 Discovered new category: {category}")
    
    # Check for new moods - FIXED logic  
    all_base_moods = []
    for moods in self.base_categories.values():
        all_base_moods.extend(moods)
    
    if mood and mood not in all_base_moods:
        if mood not in self.discovered_categories['new_moods']:
            self.discovered_categories['new_moods'].append(mood)
            print(f"🆕 Discovered new mood: {mood}")
    
    # Update frequency counts
    if category and mood:
        self.discovered_categories['frequency_counts'][f"{category}+{mood}"] += 1
    
    # Save learning data
    try:
        self.save_learning_data()
        self.save_discovered_categories()
    except Exception as e:
        print(f"Warning: Could not save learning data: {e}")
    
    # Rebuild folder map with new discoveries
    self.folder_map = self.build_dynamic_folder_map()

def fixed_generate_enhanced_filename(self, file_path, classification, audio_analysis):
    """Generate filename - FIXED for path encoding issues"""
    
    # Extract key info
    category = classification.get('category', 'unknown')
    mood = classification.get('mood', 'unknown')
    intensity = classification.get('intensity', 'unknown')
    energy = classification.get('energy_level', 0)
    
    # Get audio characteristics
    bpm = None
    brightness = None
    texture = None
    
    if audio_analysis:
        raw_features = audio_analysis.get('raw_features', {})
        interpretation = audio_analysis.get('interpretation', {})
        
        bpm = raw_features.get('tempo', 0)
        brightness = interpretation.get('brightness', '')
        texture = interpretation.get('texture', '')
    
    # Build filename components
    components = []
    
    # 1. Category prefix
    if category.startswith('music_'):
        components.append('MUS')
    elif category.startswith('sfx_'):
        sfx_type = category.replace('sfx_', '').upper()
        components.append(f'SFX_{sfx_type[:4]}')
    elif category.startswith('voice_'):
        components.append('VOX')
    else:
        components.append(category.upper()[:6])
    
    # 2. BPM (if detected and makes sense)
    if bpm and bpm > 30 and bpm < 300:
        components.append(f"{int(bpm)}bpm")
    
    # 3. Mood/Energy
    mood_short = mood[:4].upper() if mood else ""
    if mood_short:
        components.append(mood_short)
    
    # 4. Energy level
    if energy:
        components.append(f"E{energy}")
    
    # 5. Audio characteristics
    if brightness in ['bright', 'dark']:
        components.append(brightness[:3].upper())
    
    if texture == 'tonal':
        components.append('TON')
    elif texture == 'noisy':
        components.append('NOI')
    
    # 6. Intensity indicator
    intensity_map = {
        'subtle_background': 'BG',
        'narrative_support': 'SUP', 
        'dramatic_punctuation': 'DRA'
    }
    if intensity in intensity_map:
        components.append(intensity_map[intensity])
    
    # Join components
    new_filename = '_'.join(components)
    
    # Keep original extension
    original_ext = file_path.suffix
    
    # Add uniqueness - FIXED encoding issue
    file_hash = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()[:4]
    
    final_filename = f"{new_filename}_{file_hash}{original_ext}"
    
    return final_filename

# Apply the fixes
FixedAdaptiveAudioOrganizer.learn_from_classification = fixed_learn_from_classification
FixedAdaptiveAudioOrganizer.generate_enhanced_filename = fixed_generate_enhanced_filename

print("🔧 Applied critical fixes!")
print("✅ Fixed category discovery logic")
print("✅ Fixed file path encoding issue")
# Cell: Restart sweep with fixes
print("🔄 RESTARTING SWEEP WITH FIXES APPLIED")
print("Starting from file 93 where we left off...")

# Continue processing from where we stopped
remaining_files = all_audio_files[92:]  # Start from file 93

processed_count = 5  # We had 5 successful before
failed_count = 87    # We had 87 failures before
discovered_bpms = [92, 103, 78, 117]  # BPMs we found

for i, file_path in enumerate(remaining_files):
    try:
        file_num = i + 93
        print(f"\n[{file_num}/{len(all_audio_files)}]")
        
        result = organizer.process_file(file_path, dry_run=True)
        
        if result:
            processed_count += 1
            print(f"  ✅ SUCCESS!")
            
            # Track BPM
            if 'audio_analysis' in result:
                bpm = result['audio_analysis']['raw_features'].get('tempo', 0)
                if 30 < bpm < 300:
                    discovered_bpms.append(bpm)
                    print(f"  🎵 BPM: {bpm:.0f}")
        else:
            failed_count += 1
        
        # Progress update every 25 files
        if (file_num) % 25 == 0:
            print(f"\n📈 PROGRESS UPDATE:")
            print(f"  ✅ Processed: {processed_count}")
            print(f"  ❌ Failed: {failed_count}")
            print(f"  🎵 BPMs detected: {len(discovered_bpms)}")
            if discovered_bpms:
                print(f"  📊 Average BPM: {sum(discovered_bpms)/len(discovered_bpms):.0f}")
        
        time.sleep(0.3)
        
    except KeyboardInterrupt:
        print(f"\n⏹️  STOPPED by user")
        break
    except Exception as e:
        print(f"❌ Error: {e}")
        failed_count += 1

print(f"\n🎉 Current totals:")
print(f"✅ Processed: {processed_count}")
print(f"❌ Failed: {failed_count}")
print(f"🎵 BPMs detected: {len(discovered_bpms)}")
# Cell: Enhanced filename generation with original semantics
def semantic_enhanced_filename(self, file_path, classification, audio_analysis):
    """Generate filename keeping original semantic info + enhancements"""
    
    # Extract original semantic parts
    original_name = file_path.stem  # filename without extension
    
    # Clean up original name but keep meaningful parts
    semantic_parts = []
    
    # Try to extract meaningful words from original filename
    import re
    
    # Split by common separators and clean
    words = re.split(r'[_\-\s\.]+', original_name.lower())
    
    # Keep meaningful words (filter out common junk)
    meaningful_words = []
    skip_words = {'es', 'soundbed', 'instrumental', 'version', 'gen', 'sp100', 's50', 'sb75', 'se0'}
    
    for word in words:
        if (len(word) > 2 and 
            word not in skip_words and 
            not word.isdigit() and 
            not re.match(r'^\d+bpm$', word)):
            meaningful_words.append(word)
    
    # Take first 2-3 most meaningful words
    if meaningful_words:
        semantic_part = '_'.join(meaningful_words[:3])
    else:
        semantic_part = original_name[:20]  # Fallback to first 20 chars
    
    # Get classification info
    category = classification.get('category', 'unknown')
    mood = classification.get('mood', 'unknown')
    energy = classification.get('energy_level', 0)
    
    # Get BPM if available
    bpm = None
    if audio_analysis:
        raw_features = audio_analysis.get('raw_features', {})
        bpm = raw_features.get('tempo', 0)
    
    # Build enhanced filename: SEMANTIC + CLASSIFICATION + BPM + ENERGY
    components = []
    
    # 1. Semantic part first (most important)
    components.append(semantic_part)
    
    # 2. Category prefix
    if category.startswith('music_'):
        components.append('MUS')
    elif category.startswith('sfx_'):
        sfx_type = category.replace('sfx_', '').upper()
        components.append(f'SFX_{sfx_type[:4]}')
    elif category.startswith('voice_'):
        components.append('VOX')
    
    # 3. BPM if detected
    if bpm and 30 < bpm < 300:
        components.append(f"{int(bpm)}bpm")
    
    # 4. Mood + Energy
    if mood and mood != 'unknown':
        mood_short = mood[:4].upper()
        components.append(f"{mood_short}_E{energy}")
    
    # Join with underscores
    new_filename = '_'.join(components)
    
    # Add original extension
    final_filename = f"{new_filename}{file_path.suffix}"
    
    return final_filename

# Replace the method
FixedAdaptiveAudioOrganizer.generate_enhanced_filename = semantic_enhanced_filename

print("🏷️ Updated filename generation to preserve semantic meaning!")
# Cell: Debug the classification issue
def debug_classify_audio_file(self, file_path, user_description=""):
    """Debug version to see what's going wrong"""
    
    try:
        # ... (same setup as before)
        metadata = self.get_audio_metadata(file_path)
        audio_analysis = self.analyze_audio_content(file_path)
        
        # Simple test call to OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Return this exact JSON: {\"category\": \"music_ambient\", \"mood\": \"contemplative\", \"intensity\": \"subtle_background\", \"energy_level\": 5}"}],
            temperature=0.3
        )
        
        raw_response = response.choices[0].message.content.strip()
        print(f"  🔍 Raw OpenAI response: {raw_response[:100]}...")
        
        # Try to parse
        if raw_response.startswith('```json'):
            raw_response = raw_response.replace('```json', '').replace('```', '').strip()
        
        classification = json.loads(raw_response)
        print(f"  ✅ JSON parsed successfully: {classification}")
        
        return classification
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON parsing failed: {e}")
        print(f"  📄 Full response was: {raw_response}")
        return None
    except Exception as e:
        print(f"  ❌ Other error: {e}")
        return None

# Test the debug version
print("🔍 Testing OpenAI classification...")
test_file = resolve_test_file()
debug_result = debug_classify_audio_file(organizer, test_file)
# Cell: Create comprehensive metadata spreadsheet
import pandas as pd
from datetime import datetime

def create_metadata_spreadsheet(self):
    """Create a comprehensive spreadsheet of all processed files"""
    
    data = []
    
    for entry in self.learning_data['classifications']:
        classification = entry.get('classification', {})
        audio_analysis = classification.get('audio_analysis', {})
        
        # Extract all the metadata
        row = {
            # Original file info
            'original_filename': entry.get('filename', ''),
            'original_path': entry.get('file_path', ''),
            'processed_date': entry.get('timestamp', ''),
            
            # Classification data
            'category': classification.get('category', ''),
            'mood': classification.get('mood', ''),
            'intensity': classification.get('intensity', ''),
            'energy_level': classification.get('energy_level', 0),
            'confidence': classification.get('confidence', 0),
            'suggested_filename': classification.get('suggested_filename', ''),
            'target_folder': self.determine_target_folder(classification),
            
            # Audio analysis data
            'bpm': None,
            'brightness': '',
            'texture': '',
            'audio_energy': '',
            'pace': '',
            'rhythmic_activity': '',
            'spectral_centroid': None,
            'zero_crossing_rate': None,
            'rms_energy': None,
            
            # Metadata
            'tags': ', '.join(classification.get('tags', [])),
            'thematic_notes': classification.get('thematic_notes', ''),
            'reasoning': classification.get('reasoning', ''),
            'discovered_elements': ', '.join(classification.get('discovered_elements', [])),
            'audio_insights': classification.get('audio_insights', '')
        }
        
        # Add audio analysis details if available
        if audio_analysis:
            raw_features = audio_analysis.get('raw_features', {})
            interpretation = audio_analysis.get('interpretation', {})
            
            row.update({
                'bpm': raw_features.get('tempo', None),
                'brightness': interpretation.get('brightness', ''),
                'texture': interpretation.get('texture', ''),
                'audio_energy': interpretation.get('energy', ''),
                'pace': interpretation.get('pace', ''),
                'rhythmic_activity': interpretation.get('rhythmic_activity', ''),
                'spectral_centroid': raw_features.get('spectral_centroid_mean', None),
                'zero_crossing_rate': raw_features.get('zero_crossing_rate', None),
                'rms_energy': raw_features.get('rms_energy', None)
            })
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel file
    output_file = self.base_dir / "04_METADATA_SYSTEM" / f"audio_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_excel(output_file, index=False)
    
    print(f"📊 Metadata spreadsheet created: {output_file}")
    print(f"📈 Contains {len(df)} files with full analysis data")
    
    return df, output_file

# Add method to organizer
FixedAdaptiveAudioOrganizer.create_metadata_spreadsheet = create_metadata_spreadsheet

# Generate the spreadsheet
print("📊 Creating metadata spreadsheet...")
df, spreadsheet_path = organizer.create_metadata_spreadsheet()

# Show preview
print(f"\n📋 SPREADSHEET PREVIEW:")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df[['original_filename', 'suggested_filename', 'bpm', 'category', 'mood', 'energy_level']].head())

# pip install openpyxl

# Cell: Test semantic word extraction
print("\n🔍 Testing semantic word extraction...")
print("-" * 40)

test_names = [
    "88bpm playful childlike beat ES_February Moon (Instrumental Version) - Victor Lundberg",
    "90 bpm pulsing signals from digital space ES_Rhythmania 2 - August Wilhelmsson",
    "ElevenLabs_2025-05-14T00_58_03_japan presentation_gen_sp100_s50_sb75_se0_b_m2",
    "UK-Asian Attractive Young Female Accent Voice 35",
    "out-of-breath-male-176715"
]

import re

for name in test_names:
    print(f"\nOriginal: {name}")
    
    # Extract meaningful words (same logic as in the function)
    words = re.split(r'[_\-\s\.]+', name.lower())
    skip_words = {'es', 'soundbed', 'instrumental', 'version', 'gen', 'sp100', 's50', 'sb75', 'se0', 'b', 'm2'}
    
    meaningful_words = []
    for word in words:
        if (len(word) > 2 and 
            word not in skip_words and 
            not word.isdigit() and 
            not re.match(r'^\d+bpm$', word)):
            meaningful_words.append(word)
    
    semantic_part = '_'.join(meaningful_words[:3])
    print(f"Extracted: {semantic_part}")
# Cell: Create CSV version of metadata (works without openpyxl)
def create_metadata_csv(self):
    """Create CSV version of metadata (no extra libraries needed)"""
    
    data = []
    
    for entry in self.learning_data['classifications']:
        classification = entry.get('classification', {})
        audio_analysis = classification.get('audio_analysis', {})
        
        # Extract metadata
        row = {
            'original_filename': entry.get('filename', ''),
            'original_path': entry.get('file_path', ''),
            'processed_date': entry.get('timestamp', ''),
            'category': classification.get('category', ''),
            'mood': classification.get('mood', ''),
            'intensity': classification.get('intensity', ''),
            'energy_level': classification.get('energy_level', 0),
            'confidence': classification.get('confidence', 0),
            'suggested_filename': classification.get('suggested_filename', ''),
            'target_folder': self.determine_target_folder(classification),
            'tags': ', '.join(classification.get('tags', [])),
            'thematic_notes': classification.get('thematic_notes', ''),
            'reasoning': classification.get('reasoning', ''),
            'audio_insights': classification.get('audio_insights', '')
        }
        
        # Add audio analysis if available
        if audio_analysis:
            raw_features = audio_analysis.get('raw_features', {})
            interpretation = audio_analysis.get('interpretation', {})
            
            row.update({
                'bpm': raw_features.get('tempo', ''),
                'brightness': interpretation.get('brightness', ''),
                'texture': interpretation.get('texture', ''),
                'audio_energy': interpretation.get('energy', ''),
                'pace': interpretation.get('pace', ''),
                'rhythmic_activity': interpretation.get('rhythmic_activity', '')
            })
        
        data.append(row)
    
    # Create DataFrame and save as CSV
    import pandas as pd
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_file = self.base_dir / "04_METADATA_SYSTEM" / f"audio_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(csv_file, index=False)
    
    print(f"📄 CSV metadata created: {csv_file}")
    print(f"📈 Contains {len(df)} files with analysis data")
    
    return df, csv_file

# Add CSV method
FixedAdaptiveAudioOrganizer.create_metadata_csv = create_metadata_csv

# Create CSV version
print("📄 Creating CSV metadata...")
df, csv_path = organizer.create_metadata_csv()

# Show what we got
print(f"\n📊 METADATA SUMMARY:")
print(f"Total files analyzed: {len(df)}")
print(f"Categories discovered: {df['category'].nunique()}")
print(f"Moods discovered: {df['mood'].nunique()}")
print(f"Files with BPM: {df['bpm'].notna().sum()}")

if len(df) > 0:
    print(f"\n📋 SAMPLE DATA:")
    cols_to_show = ['original_filename', 'suggested_filename', 'bpm', 'category', 'mood', 'energy_level']
    print(df[cols_to_show].head(3).to_string(index=False))

print(f"\n🎯 DYNAMIC FOLDER STRUCTURE CREATED:")
print("Your system learned and created these new organizational patterns!")
# Cell: Create Excel version (after installing openpyxl)
df_excel, excel_path = organizer.create_metadata_spreadsheet()
print(f"📊 Excel file created: {excel_path}")
# Cell: PERFECTED semantic extraction - fixing the specific issues
def perfect_semantic_extraction(self, original_name):
    """Extract semantics with PERFECT preservation - final version"""
    
    import re
    
    # Clean but preserve structure
    name = original_name
    
    # Remove technical junk but keep meaningful content
    junk_patterns = [
        r'ES_',  # Remove ES_ prefix
        r'gen_sp\d+_s\d+_sb\d+_se\d+_b_m\d+',  # Remove technical generation params
        r'\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}',  # Remove timestamps
        r'_gen_sp\d+.*$',  # Remove everything after _gen_sp
    ]
    
    for pattern in junk_patterns:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    
    # Remove BPM since librosa will detect it
    name = re.sub(r'\d+\s*bpm\s*', '', name, flags=re.IGNORECASE)
    
    # Determine content type more accurately
    content_type = 'unknown'
    if 'out-of-breath' in name.lower() or 'breathing' in name.lower() or 'heartbeat' in name.lower():
        content_type = 'sfx_human'  # These are sound effects, not voice samples
    elif 'voice' in name.lower() or (any(word in name.lower() for word in ['male', 'female', 'narrator', 'accent']) and 'beat' not in name.lower()):
        content_type = 'voice'
    elif any(word in name.lower() for word in ['beat', 'rhythm', 'song', 'track', 'music', 'instrumental']):
        content_type = 'music'
    elif any(word in name.lower() for word in ['sfx', 'sound', 'effect', 'ambient', 'drone', 'signals', 'digital']):
        content_type = 'sfx'
    
    if content_type == 'voice':
        return self._extract_voice_semantics_perfect(name)
    elif content_type == 'sfx_human':
        return self._extract_sfx_human_semantics_perfect(name)
    elif content_type == 'music':
        return self._extract_music_semantics_perfect(name)
    elif content_type == 'sfx':
        return self._extract_sfx_semantics_perfect(name)
    else:
        return self._extract_general_semantics_perfect(name)

def _extract_voice_semantics_perfect(self, name):
    """Extract voice semantics - PERFECT VERSION"""
    components = []
    
    # Handle compound descriptors like "UK-Asian"
    if 'uk-asian' in name.lower():
        components.append('UK-Asian')
    elif 'uk' in name.lower() or 'british' in name.lower():
        components.append('UK')
    elif 'asian' in name.lower():
        components.append('Asian')
    
    # Age/Gender as compound when possible
    age_gender = []
    if 'young' in name.lower():
        age_gender.append('Young')
    if 'female' in name.lower():
        age_gender.append('Female')
    elif 'male' in name.lower():
        age_gender.append('Male')
    
    if len(age_gender) > 1:
        components.append('_'.join(age_gender))  # "Young_Female"
    else:
        components.extend(age_gender)
    
    # Voice characteristics
    characteristics = ['attractive', 'deep', 'gruff', 'clear', 'raspy']
    for char in characteristics:
        if char.lower() in name.lower():
            components.append(char.title())
    
    # Always end with "Voice" for voice samples
    components.append('Voice')
    
    # Numbers (age, version) - FIXED to include
    numbers = re.findall(r'\b(\d{2,3})\b', name)
    if numbers:
        components.append(numbers[0])
    
    return '_'.join(components)

def _extract_sfx_human_semantics_perfect(self, name):
    """Extract human SFX semantics - PERFECT VERSION"""
    components = []
    
    # Gender
    if 'female' in name.lower():
        components.append('Female')
    elif 'male' in name.lower():
        components.append('Male')
    
    # Main descriptor - FIXED to detect "out-of-breath"
    if 'out-of-breath' in name.lower() or 'out_of_breath' in name.lower():
        components.append('Out-of-Breath')
    elif 'breathing' in name.lower():
        components.append('Breathing')
    elif 'heartbeat' in name.lower():
        components.append('Heartbeat')
    
    # Always end with SFX for sound effects - FIXED
    components.append('SFX')
    
    # Numbers
    numbers = re.findall(r'\b(\d{3,6})\b', name)
    if numbers:
        components.append(numbers[0])
    
    return '_'.join(components)

def _extract_music_semantics_perfect(self, name):
    """Extract music semantics - PERFECT VERSION"""
    components = []
    
    # Remove common prefixes
    name = re.sub(r'^es_|^soundbed', '', name, flags=re.IGNORECASE)
    
    # Extract user descriptors first
    descriptive_words = [
        'playful', 'childlike', 'pulsing', 'signals', 'digital', 'space', 'clinical', 
        'ambient', 'atmospheric', 'cinematic', 'touching', 'magical', 'hopeful',
        'dark', 'bright', 'mysterious', 'contemplative', 'melancholic', 'tense'
    ]
    
    words = re.split(r'[_\s\-]+', name)
    descriptors = []
    
    # Collect descriptors in order
    for word in words:
        if word.lower() in [d.lower() for d in descriptive_words]:
            descriptors.append(word.lower())
    
    # Add descriptors first
    components.extend(descriptors)
    
    # Extract title and artist - FIXED to preserve full title
    artist_match = re.search(r'\s*-\s*([^-]+?)(?:\.|$)', name)
    if artist_match:
        title_part = name[:artist_match.start()].strip()
        artist_part = artist_match.group(1).strip()
        
        # Clean title part - FIXED to preserve parentheses content
        title_clean = re.sub(r'^(es[\s_]*|soundbed[\s_]*)', '', title_part, flags=re.IGNORECASE)
        # Remove descriptors from title
        for desc in descriptors:
            title_clean = re.sub(rf'\b{re.escape(desc)}\b', '', title_clean, flags=re.IGNORECASE)
        
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()  # Clean up spaces
        if title_clean:
            components.append(title_clean)
        
        # Add artist name - FIXED to keep first name
        artist_words = artist_part.split()
        if artist_words:
            components.append(artist_words[0])
    else:
        # No clear separation, take meaningful words excluding descriptors
        other_words = [w for w in words if w.lower() not in descriptors and len(w) > 1 and not w.isdigit()]
        if other_words:
            components.extend(other_words[:2])
    
    return '_'.join(components)

def _extract_sfx_semantics_perfect(self, name):
    """Extract SFX semantics - PERFECT VERSION"""
    components = []
    
    # Key SFX descriptors - FIXED to include "space" and filter "from"
    descriptors = ['pulsing', 'signals', 'digital', 'space', 'mechanical', 'ambient', 'atmospheric']
    skip_words = {'es', 'from', 'the', 'and', 'or', 'a', 'an'}
    
    words = re.split(r'[_\s\-]+', name)
    
    for word in words:
        if word.lower() in descriptors:
            components.append(word.lower())
        elif len(word) > 2 and not word.isdigit() and word.lower() not in skip_words:
            if len(components) < 4:  # Allow more components
                components.append(word)
    
    return '_'.join(components[:4])

def _extract_general_semantics_perfect(self, name):
    """General extraction - PERFECT VERSION"""
    words = re.split(r'[_\s\-]+', name)
    skip_words = {'es', 'soundbed', 'instrumental', 'version', 'gen', 'from', 'the'}
    
    meaningful = []
    for word in words:
        if (len(word) > 1 and 
            word.lower() not in skip_words and 
            not word.isdigit()):
            meaningful.append(word)
        
        if len(meaningful) >= 3:
            break
    
    return '_'.join(meaningful[:3])

# Replace all methods with PERFECT versions
FixedAdaptiveAudioOrganizer.enhanced_semantic_extraction = perfect_semantic_extraction
FixedAdaptiveAudioOrganizer._extract_voice_semantics_perfect = _extract_voice_semantics_perfect
FixedAdaptiveAudioOrganizer._extract_sfx_human_semantics_perfect = _extract_sfx_human_semantics_perfect
FixedAdaptiveAudioOrganizer._extract_music_semantics_perfect = _extract_music_semantics_perfect
FixedAdaptiveAudioOrganizer._extract_sfx_semantics_perfect = _extract_sfx_semantics_perfect
FixedAdaptiveAudioOrganizer._extract_general_semantics_perfect = _extract_general_semantics_perfect

print("🎯 PERFECT semantic extraction implemented!")
# Cell: Interactive classification with user prompts
def interactive_classify_audio_file(self, file_path, batch_mode=False):
    """Classify with optional user interaction for better accuracy"""
    
    metadata = self.get_audio_metadata(file_path)
    audio_analysis = self.analyze_audio_content(file_path)
    
    # Check if we should ask for user input
    ask_for_input = False
    confidence_threshold = 0.7
    
    # Initial AI classification
    initial_classification = self._get_initial_ai_classification(file_path, metadata, audio_analysis)
    
    if initial_classification:
        confidence = initial_classification.get('confidence', 0)
        if confidence < confidence_threshold:
            ask_for_input = True
    else:
        ask_for_input = True
    
    # Interactive prompts when needed
    if ask_for_input and not batch_mode:
        print(f"\n🤔 I need some help with: {file_path.name}")
        
        if audio_analysis:
            bpm = audio_analysis['raw_features'].get('tempo', 0)
            interpretation = audio_analysis['interpretation']
            print(f"🎵 Audio analysis: {bpm:.0f} BPM, {interpretation.get('brightness', '')}, {interpretation.get('energy', '')} energy")
        
        # Ask targeted questions
        user_input = self._ask_classification_questions(file_path, initial_classification)
        
        # Incorporate user feedback
        enhanced_classification = self._enhance_with_user_input(initial_classification, user_input)
        
        return enhanced_classification
    
    return initial_classification

def _ask_classification_questions(self, file_path, initial_classification):
    """Ask targeted questions to improve classification"""
    
    user_input = {}
    
    print(f"📝 Quick questions to improve classification:")
    
    # Question 1: Content type if unclear
    if not initial_classification or initial_classification.get('confidence', 0) < 0.5:
        print(f"\n1. What type of content is this?")
        print(f"   a) Music/Musical piece")
        print(f"   b) Sound effect/SFX") 
        print(f"   c) Voice/Speech")
        print(f"   d) Ambient/Atmospheric")
        
        content_type = input("   Your choice (a/b/c/d or description): ").strip().lower()
        user_input['content_type'] = content_type
    
    # Question 2: Mood/feeling
    print(f"\n2. What mood or feeling does this convey?")
    print(f"   (e.g., mysterious, contemplative, tense, playful, etc.)")
    mood_input = input("   Mood: ").strip()
    if mood_input:
        user_input['mood'] = mood_input
    
    # Question 3: How would you use this?
    print(f"\n3. How would you use this in your AI consciousness project?")
    print(f"   a) Background atmosphere")
    print(f"   b) Supporting narrative moments") 
    print(f"   c) Dramatic emphasis/punctuation")
    
    usage = input("   Usage (a/b/c or description): ").strip()
    if usage:
        user_input['usage'] = usage
    
    # Question 4: Any specific tags or themes?
    print(f"\n4. Any specific themes or tags? (consciousness, transformation, memory, etc.)")
    tags_input = input("   Tags: ").strip()
    if tags_input:
        user_input['tags'] = [tag.strip() for tag in tags_input.split(',')]
    
    print(f"   Thanks! Processing with your input... 🎯")
    
    return user_input

def _enhance_with_user_input(self, ai_classification, user_input):
    """Enhance AI classification with user input"""
    
    enhanced = ai_classification.copy() if ai_classification else {}
    
    # Map user content type to our categories
    content_mapping = {
        'a': 'music_ambient',
        'music': 'music_ambient',
        'musical': 'music_ambient',
        'b': 'sfx_environmental', 
        'sfx': 'sfx_environmental',
        'sound': 'sfx_environmental',
        'c': 'voice_element',
        'voice': 'voice_element',
        'speech': 'voice_element',
        'd': 'music_ambient',
        'ambient': 'music_ambient'
    }
    
    # Apply user inputs
    if 'content_type' in user_input:
        content = user_input['content_type']
        if content in content_mapping:
            enhanced['category'] = content_mapping[content]
        else:
            # Try to map custom descriptions
            if 'music' in content or 'song' in content:
                enhanced['category'] = 'music_ambient'
            elif 'voice' in content or 'speech' in content:
                enhanced['category'] = 'voice_element'
            elif 'sfx' in content or 'effect' in content:
                enhanced['category'] = 'sfx_environmental'
    
    if 'mood' in user_input:
        enhanced['mood'] = user_input['mood']
    
    # Map usage to intensity
    usage_mapping = {
        'a': 'subtle_background',
        'background': 'subtle_background',
        'b': 'narrative_support',
        'narrative': 'narrative_support', 
        'support': 'narrative_support',
        'c': 'dramatic_punctuation',
        'dramatic': 'dramatic_punctuation',
        'emphasis': 'dramatic_punctuation'
    }
    
    if 'usage' in user_input:
        usage = user_input['usage']
        if usage in usage_mapping:
            enhanced['intensity'] = usage_mapping[usage]
    
    if 'tags' in user_input:
        existing_tags = enhanced.get('tags', [])
        enhanced['tags'] = existing_tags + user_input['tags']
    
    # Boost confidence since user provided input
    enhanced['confidence'] = 0.95
    enhanced['reasoning'] = f"Enhanced with user input: {user_input}"
    
    return enhanced

# Add methods to organizer
FixedAdaptiveAudioOrganizer.interactive_classify_audio_file = interactive_classify_audio_file
FixedAdaptiveAudioOrganizer._ask_classification_questions = _ask_classification_questions
FixedAdaptiveAudioOrganizer._enhance_with_user_input = _enhance_with_user_input

# Keep the original method as backup
FixedAdaptiveAudioOrganizer._get_initial_ai_classification = FixedAdaptiveAudioOrganizer.classify_audio_file

print("🤔 Interactive classification system added!")
# Cell: Enable interactive mode and test it
print("🤔 Setting up interactive classification mode...")

# First, let's test with a file that might be uncertain
test_file = resolve_test_file()

if test_file.exists():
    print(f"🎯 Testing interactive mode with: {test_file.name}")
    print("This will ask you questions if the AI needs help!")
    print("-" * 60)
    
    # Test interactive classification
    result = organizer.process_file_interactive(test_file, batch_mode=False, dry_run=True)
    
    if result:
        print(f"\n🎉 Interactive classification complete!")
        print(f"Final confidence: {result.get('confidence', 0):.1%}")
    else:
        print(f"\n❌ Classification failed even with interaction")
else:
    print(f"❌ Test file not found: {test_file}")
    
    # Let's find any file to test with
    print("🔍 Looking for any audio file to test with...")
    
    # Check TO_SORT folder
    to_sort = BASE_DIRECTORY / "TO_SORT"
    if to_sort.exists():
        audio_files = [f for f in to_sort.iterdir() 
                      if f.is_file() and f.suffix.lower() in {'.mp3', '.wav', '.m4a', '.aiff'}]
        
        if audio_files:
            test_file = audio_files[0]
            print(f"✅ Found test file: {test_file.name}")
            
            result = organizer.process_file_interactive(test_file, batch_mode=False, dry_run=True)
        else:
            print("❌ No audio files found in TO_SORT")
# Cell: Interactive batch processing with selective prompting
def interactive_batch_process(self, file_paths, confidence_threshold=0.7):
    """Process multiple files with interactive mode for uncertain ones"""
    
    print(f"🚀 Starting interactive batch processing...")
    print(f"📊 Will ask for help when confidence < {confidence_threshold:.0%}")
    print(f"📁 Processing {len(file_paths)} files")
    print("=" * 60)
    
    results = []
    high_confidence_count = 0
    interactive_count = 0
    
    for i, file_path in enumerate(file_paths[:5]):  # Limit to first 5 for testing
        print(f"\n[{i+1}/{min(len(file_paths), 5)}]")
        
        # Get initial AI classification to check confidence
        metadata = self.get_audio_metadata(file_path)
        audio_analysis = self.analyze_audio_content(file_path)
        initial_classification = self._get_initial_ai_classification(file_path, metadata, audio_analysis)
        
        if initial_classification and initial_classification.get('confidence', 0) >= confidence_threshold:
            # High confidence - process automatically
            print(f"🤖 High confidence - processing automatically")
            result = self.process_file_interactive(file_path, batch_mode=True, dry_run=True)
            high_confidence_count += 1
        else:
            # Low confidence - ask for help
            print(f"🤔 Low confidence - requesting user input")
            result = self.process_file_interactive(file_path, batch_mode=False, dry_run=True)
            interactive_count += 1
        
        if result:
            results.append(result)
        
        print("-" * 30)
    
    print(f"\n📊 BATCH PROCESSING SUMMARY:")
    print(f"✅ Total processed: {len(results)}")
    print(f"🤖 Auto-processed (high confidence): {high_confidence_count}")
    print(f"🤔 Interactive (needed help): {interactive_count}")
    print(f"📈 Success rate: {len(results)/min(len(file_paths), 5):.1%}")
    
    return results

# Add to organizer
FixedAdaptiveAudioOrganizer.interactive_batch_process = interactive_batch_process

# Test interactive batch processing
print("\n🚀 Testing interactive batch processing...")

# Get some files to test with
to_sort_path = BASE_DIRECTORY / "TO_SORT"
if to_sort_path.exists():
    audio_files = [f for f in to_sort_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.mp3', '.wav', '.m4a', '.aiff'}]
    
    if audio_files:
        print(f"Found {len(audio_files)} files to test with")
        
        # Run interactive batch processing
        results = organizer.interactive_batch_process(audio_files[:3])  # Test with first 3
    else:
        print("No audio files found for testing")
else:
    print("TO_SORT folder not found")

# Cell: Fix batch processing and add missing method
def interactive_batch_process(self, file_paths, confidence_threshold=0.7):
    """Process multiple files with interactive mode for uncertain ones"""
    
    print(f"🚀 Starting interactive batch processing...")
    print(f"📊 Will ask for help when confidence < {confidence_threshold:.0%}")
    print(f"📁 Processing {len(file_paths)} files")
    print("=" * 60)
    
    results = []
    high_confidence_count = 0
    interactive_count = 0
    
    for i, file_path in enumerate(file_paths[:5]):  # Limit to first 5 for testing
        print(f"\n[{i+1}/{min(len(file_paths), 5)}]")
        
        # Get initial AI classification to check confidence
        # Use the regular classify method since we don't have _get_initial_ai_classification
        initial_classification = self.classify_audio_file(file_path)
        
        if initial_classification and initial_classification.get('confidence', 0) >= confidence_threshold:
            # High confidence - process automatically
            print(f"🤖 High confidence ({initial_classification.get('confidence', 0):.1%}) - processing automatically")
            result = self.process_file_interactive(file_path, batch_mode=True, dry_run=True)
            high_confidence_count += 1
        else:
            # Low confidence - ask for help
            confidence = initial_classification.get('confidence', 0) if initial_classification else 0
            print(f"🤔 Low confidence ({confidence:.1%}) - requesting user input")
            result = self.process_file_interactive(file_path, batch_mode=False, dry_run=True)
            interactive_count += 1
        
        if result:
            results.append(result)
        
        print("-" * 30)
    
    print(f"\n📊 BATCH PROCESSING SUMMARY:")
    print(f"✅ Total processed: {len(results)}")
    print(f"🤖 Auto-processed (high confidence): {high_confidence_count}")
    print(f"🤔 Interactive (needed help): {interactive_count}")
    print(f"📈 Success rate: {len(results)/min(len(file_paths), 5):.1%}")
    
    return results

# Add the fixed batch processing method
organizer.interactive_batch_process = interactive_batch_process.__get__(organizer, FixedAdaptiveAudioOrganizer)

print("🔧 Fixed batch processing method!")
# Cell: Test the fixed batch processing
to_sort_path = BASE_DIRECTORY / "TO_SORT"

if to_sort_path.exists():
    audio_files = [f for f in to_sort_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.mp3', '.wav', '.m4a', '.aiff'}]
    
    if audio_files:
        print(f"🎯 Found {len(audio_files)} files to test with")
        print("🚀 Starting corrected interactive batch processing...")
        
        # Run interactive batch processing with the fixed method
        results = organizer.interactive_batch_process(audio_files[:2])  # Test with first 2
        
        if results:
            print(f"\n🎉 Batch processing complete!")
            print(f"Sample results:")
            for result in results[:2]:
                filename = result.get('suggested_filename', 'No filename')
                confidence = result.get('confidence', 0)
                print(f"  📄 {filename} (confidence: {confidence:.1%})")
    else:
        print("No audio files found for testing")
else:
    print("TO_SORT folder not found")
# Cell: Test with a file that will definitely trigger interactive mode
print("🎯 Let's find a file that will make the AI uncertain...")

# Force the system to ask questions by lowering the threshold
organizer.set_interaction_mode('minimal')  # Only ask when very uncertain (40% threshold)

# Or we can temporarily force it to always ask
print("🤔 Setting to 'always ask' mode to demonstrate the interactive questions...")
organizer.set_interaction_mode('always')

# Test with the same file - now it will definitely ask questions
test_file = resolve_test_file()

if test_file.exists():
    print(f"📁 Testing interactive mode with: {test_file.name}")
    print("This time it WILL ask questions regardless of confidence...")
    print("-" * 60)
    
    result = organizer.process_file_interactive(test_file, batch_mode=False, dry_run=True)
    
    if result:
        print(f"\n🎉 Interactive classification complete!")
        print(f"Your input boosted confidence to: {result.get('confidence', 0):.1%}")
        print(f"Enhanced with your feedback: {result.get('reasoning', '')}")
else:
    print("❌ Test file not found")

# Reset to smart mode after testing
organizer.set_interaction_mode('smart')
print(f"\n🔄 Reset to smart mode for normal operation")
# Cell: Add audio playback to interactive classification
import IPython.display as ipd
from pathlib import Path

def play_audio_clip(self, file_path, duration=10):
    """Play a short clip of the audio file for classification help"""
    try:
        print(f"🎵 Playing {duration}-second preview of: {file_path.name}")
        
        # For Jupyter notebooks, we can use IPython's Audio display
        # This will create a player widget
        audio_widget = ipd.Audio(str(file_path), autoplay=False)
        
        # Display the audio player
        ipd.display(audio_widget)
        
        print("🎧 Use the player above to listen while answering questions!")
        return True
        
    except Exception as e:
        print(f"❌ Could not play audio: {e}")
        print("💡 Try opening the file manually to listen while classifying")
        return False

def enhanced_ask_classification_questions(self, file_path, initial_classification):
    """Ask questions with audio playback support"""
    
    print(f"\n📝 Quick questions to improve classification:")
    
    # Play audio clip first!
    print(f"\n🎵 Let's listen to the audio first...")
    self.play_audio_clip(file_path, duration=15)
    
    user_input = {}
    
    # Question 1: Content type if unclear
    if not initial_classification or initial_classification.get('confidence', 0) < 0.5:
        print(f"\n1. After listening, what type of content is this?")
        print(f"   a) Music/Musical piece")
        print(f"   b) Sound effect/SFX") 
        print(f"   c) Voice/Speech")
        print(f"   d) Ambient/Atmospheric")
        
        content_type = input("   Your choice (a/b/c/d or description): ").strip().lower()
        user_input['content_type'] = content_type
    
    # Question 2: Mood/feeling (now with audio context!)
    print(f"\n2. After listening, what mood or feeling does this convey?")
    print(f"   (e.g., mysterious, contemplative, tense, playful, eerie, hopeful, etc.)")
    mood_input = input("   Mood: ").strip()
    if mood_input:
        user_input['mood'] = mood_input
    
    # Question 3: How would you use this?
    print(f"\n3. How would you use this in your AI consciousness project?")
    print(f"   a) Background atmosphere")
    print(f"   b) Supporting narrative moments") 
    print(f"   c) Dramatic emphasis/punctuation")
    
    usage = input("   Usage (a/b/c or description): ").strip()
    if usage:
        user_input['usage'] = usage
    
    # Question 4: Any specific tags or themes?
    print(f"\n4. Any specific themes this evokes? (consciousness, memory, digital, organic, etc.)")
    tags_input = input("   Tags: ").strip()
    if tags_input:
        user_input['tags'] = [tag.strip() for tag in tags_input.split(',')]
    
    # Question 5: Semantic description for filename
    print(f"\n5. How would you describe this for the filename?")
    print(f"   (e.g., 'eerie digital pulse', 'warm contemplative pad', 'glitchy consciousness')")
    semantic_input = input("   Description: ").strip()
    if semantic_input:
        user_input['semantic_description'] = semantic_input
    
    print(f"\n   Thanks! That audio context really helps! 🎯")
    
    return user_input

# Add audio playback methods
organizer.play_audio_clip = play_audio_clip.__get__(organizer, FixedAdaptiveAudioOrganizer)

# Replace the questioning method with the enhanced version
organizer._ask_classification_questions = enhanced_ask_classification_questions.__get__(organizer, FixedAdaptiveAudioOrganizer)

print("🎵 Audio playback added to interactive classification!")
# Cell: Test interactive classification with audio playback
test_file = resolve_test_file()

print("🎵 Testing interactive classification WITH audio playback!")
print("You'll be able to listen while answering questions!")
print("-" * 60)

if test_file.exists():
    # Force interactive mode
    organizer.set_interaction_mode('always')
    
    result = organizer.process_file_interactive(test_file, batch_mode=False, dry_run=True)
    
    if result:
        print(f"\n🎉 Audio-enhanced classification complete!")
        print(f"Enhanced filename: {result.get('suggested_filename', 'None')}")
        
        # Show the improvement
        print(f"\n📈 BEFORE vs AFTER:")
        print(f"🤖 AI only: sfx_environmental + contemplative (85% confidence)")
        print(f"🎵 With audio + your input: {result.get('category', '')} + {result.get('mood', '')} ({result.get('confidence', 0):.1%} confidence)")
    else:
        print("❌ Classification failed")
else:
    print("❌ Test file not found")
# Ryan's Audio Organization Notebook
# ADHD-Friendly Setup for "The Papers That Dream"

# Example: Setting up AudioAI Organization System
# This section demonstrates how to use the system with environment variables

try:
    from config import OPENAI_API_KEY, BASE_DIRECTORY, DIRECTORIES_TO_SCAN, ensure_base_structure
    ensure_base_structure()
except RuntimeError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)

print("🎵 Setting up AudioAI Organization System")
print("=" * 50)
print(f"📁 Base Directory: {BASE_DIRECTORY}")
print(f"🔍 Scan Directories: {DIRECTORIES_TO_SCAN or [BASE_DIRECTORY]}")
print()

# ================================================================
# STEP 1: Initialize the System
# ================================================================

from audioai_organizer import AdaptiveAudioOrganizer

# Initialize with base directory
organizer = AdaptiveAudioOrganizer(OPENAI_API_KEY, str(BASE_DIRECTORY))

# Add custom categories if needed
custom_categories = {
    "music_custom": ["contemplative", "energetic", "mysterious", "uplifting"],
    "sfx_custom": ["ambient", "mechanical", "digital", "organic"],
    "voice_custom": ["narrative", "dialogue", "monologue", "announcement"]
}

organizer.add_custom_categories(custom_categories)
organizer.set_interaction_mode('minimal')  # Less interruption for workflow

print("✅ AudioAI Organizer initialized!")
print()

# ================================================================
# STEP 2: Find Audio Files
# ================================================================

def find_audio_files(locations):
    """Find all audio files in the specified locations"""
    all_files = []
    
    for location in locations:
        location_path = Path(location)
        if not location_path.exists():
            print(f"⚠️ Location doesn't exist: {location}")
            continue
            
        print(f"🔍 Scanning: {location}")
        
        files = organizer.find_audio_files_recursive(str(location))
        
        print(f"  📄 Found {len(files)} audio files")
        all_files.extend(files)
        
        # Show first few examples
        for example in files[:3]:
            print(f"    - {Path(example).name}")
        if len(stragglers) > 3:
            print(f"    ... and {len(stragglers) - 3} more")
        print()
    
    return all_stragglers

print("🧹 FINDING AUDIO STRAGGLERS")
print("=" * 30)
stragglers = find_audio_stragglers(STRAGGLER_LOCATIONS)
print(f"📊 TOTAL STRAGGLERS FOUND: {len(stragglers)}")
print()

# ================================================================
# STEP 3: Organize the Stragglers (DRY RUN FIRST!)
# ================================================================

if stragglers:
    print("🔄 ORGANIZING STRAGGLERS (DRY RUN)")
    print("=" * 35)
    print("This will show you what would happen without actually moving files")
    print()
    
    # Process in dry run mode first
    results = organizer.interactive_batch_process(
        stragglers, 
        confidence_threshold=0.6,  # Ask for help on uncertain files
        dry_run=True
    )
    
    print(f"\n📋 DRY RUN COMPLETE!")
    print(f"✅ Would organize: {len([r for r in results if r])} files")
    print(f"❓ Would need input on: {len([r for r in results if not r])} files")
    print()
    print("💡 To actually move files, run:")
    print("    results = organizer.interactive_batch_process(stragglers, dry_run=False)")
else:
    print("🎉 No stragglers found! Your audio is already well-organized.")

print()

# ================================================================
# STEP 4: Attention Episode Workspace
# ================================================================

print("🎬 ATTENTION IS ALL YOU NEED - Episode Workspace")
print("=" * 50)

# Find the Attention episode folder
attention_episode = None
for folder in Path(PAPERS_PROJECT).iterdir():
    if "attention" in folder.name.lower():
        attention_episode = folder
        break

if attention_episode:
    print(f"📁 Found episode folder: {attention_episode}")
    
    # Quick analysis of existing audio in episode
    episode_audio = []
    for root, dirs, files in os.walk(attention_episode):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ['.mp3', '.wav', '.aiff', '.m4a']):
                episode_audio.append(os.path.join(root, file))
    
    print(f"🎵 Existing audio files in episode: {len(episode_audio)}")
    for audio_file in episode_audio[:5]:  # Show first 5
        print(f"  - {Path(audio_file).name}")
    if len(episode_audio) > 5:
        print(f"  ... and {len(episode_audio) - 5} more")
else:
    print("⚠️ Attention episode folder not found")
    print("Available episode folders:")
    for folder in Path(PAPERS_PROJECT).iterdir():
        if folder.is_dir() and "episode" in folder.name.lower():
            print(f"  - {folder.name}")

print()

# ================================================================
# STEP 5: Cue Suggestion Functions
# ================================================================

def suggest_cues_for_scene(scene_description, max_suggestions=5):
    """Get audio cue suggestions for a specific scene"""
    print(f"🎬 Scene: {scene_description}")
    print(f"🔍 Searching organized library...")
    
    # Search through learning data for matches
    matches = []
    scene_words = scene_description.lower().split()
    
    for classification_entry in organizer.learning_data['classifications']:
        classification = classification_entry.get('classification', {})
        filename = classification_entry.get('filename', '')
        
        # Calculate relevance score
        relevance = 0
        
        # Check mood matching
        mood = classification.get('mood', '').lower()
        for word in scene_words:
            if word in mood:
                relevance += 3
        
        # Check tags
        tags = classification.get('tags', [])
        for tag in tags:
            for word in scene_words:
                if word in tag.lower():
                    relevance += 2
        
        # Check reasoning
        reasoning = classification.get('reasoning', '').lower()
        for word in scene_words:
            if word in reasoning:
                relevance += 1
        
        # Special attention-themed matches
        if any(attention_word in scene_description.lower() for attention_word in 
               ['attention', 'focus', 'transform', 'pattern', 'neural', 'process']):
            if any(attention_word in str(classification).lower() for attention_word in 
                   ['attention', 'focus', 'neural', 'process', 'digital', 'transform']):
                relevance += 4
        
        if relevance > 0:
            matches.append({
                'filename': filename,
                'relevance': relevance,
                'mood': classification.get('mood', 'unknown'),
                'energy': classification.get('energy_level', 0),
                'intensity': classification.get('intensity', 'unknown'),
                'tags': classification.get('tags', []),
                'reasoning': classification.get('reasoning', ''),
                'confidence': classification.get('confidence', 0)
            })
    
    # Sort by relevance
    matches.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Display suggestions
    if matches:
        print(f"✨ Found {len(matches)} potential matches:")
        for i, match in enumerate(matches[:max_suggestions], 1):
            print(f"\n{i}. {match['filename']}")
            print(f"   🎭 Mood: {match['mood']} | ⚡ Energy: {match['energy']}/10")
            print(f"   🎯 Usage: {match['intensity']} | 🏷️ Tags: {', '.join(match['tags'][:3])}")
            print(f"   💭 Why: {match['reasoning'][:100]}...")
            print(f"   🔥 Confidence: {match['confidence']:.1%} | 📊 Relevance: {match['relevance']}")
    else:
        print("🤔 No matches found in organized library yet.")
        print("💡 Try organizing more audio first, or describe the scene differently!")
    
    return matches[:max_suggestions]

# Ready-to-use functions
print("🎯 QUICK FUNCTIONS READY:")
print("=" * 25)
print("suggest_cues_for_scene('quiet moment of attention mechanism awakening')")
print("suggest_cues_for_scene('intense transformation of neural patterns')")
print("suggest_cues_for_scene('human discovering AI attention breakthrough')")
print("organizer.show_learning_stats()")
print()
print("🚀 System ready! Try describing a scene from your Attention episode!")
# This will find stragglers and show you what would happen
# (Run each section of the notebook above)

suggest_cues_for_attention_scene("quiet moment of attention mechanism awakening")
# Ryan's Audio Organization - LIVE WORKFLOW
# Ready to rock with 278 files already processed! 🎉

import os
from pathlib import Path
from audioai_organizer import AdaptiveAudioOrganizer

# Example setup using config-driven approach
try:
    from config import OPENAI_API_KEY, BASE_DIRECTORY, DIRECTORIES_TO_SCAN, ensure_base_structure
    ensure_base_structure()
except RuntimeError as e:
    print(f"❌ Configuration error: {e}")
    exit(1)

# Initialize with base directory  
organizer = AdaptiveAudioOrganizer(OPENAI_API_KEY, str(BASE_DIRECTORY))

# Add custom categories for your project
custom_categories = {
    "music_custom": ["contemplative", "energetic", "mysterious", "uplifting"],
    "sfx_custom": ["ambient", "mechanical", "digital", "organic"],
    "voice_custom": ["narrative", "dialogue", "monologue", "announcement"]
}

organizer.add_custom_categories(custom_categories)
organizer.set_interaction_mode('minimal')  # Less interruption

print("🧠 Enhanced with custom themes!")
print("📊 Current learning state:")
organizer.show_learning_stats()

# ================================================================
# STEP 1: Find Audio Files
# ================================================================

print("\n" + "="*60)
print("🧹 FINDING AUDIO FILES")
print("="*60)

# Use directories from environment or fallback to base
scan_locations = DIRECTORIES_TO_SCAN if DIRECTORIES_TO_SCAN else [BASE_DIRECTORY]

def find_audio_files_smart(locations):
    """Find audio files with smart filtering"""
    all_files = []
    
    for location in locations:
        location_path = Path(location)
        if not location_path.exists():
            print(f"⚠️ Location doesn't exist: {location}")
            continue
            
        print(f"\n🔍 Scanning: {location}")
        
        stragglers = organizer.find_audio_files_recursive(location)
        
        # Smart filtering for Desktop/Documents to avoid chaos
        if any(folder in location for folder in ['Desktop', 'Documents']):
            original_count = len(stragglers)
            filtered_stragglers = []
            
            for file_path in stragglers:
                file_path_obj = Path(file_path)
                path_parts = file_path_obj.parts
                
                # Include if:
                # 1. File is not too deep (max 3 levels from Desktop/Documents)
                # 2. OR in obviously audio-related folders
                # 3. OR larger files (likely intentional audio)
                
                is_shallow = len(path_parts) <= 5
                is_audio_folder = any(audio_word in str(file_path).lower() for audio_word in 
                                    ['audio', 'music', 'sound', 'voice', 'podcast', 'episode', 'intro'])
                
                try:
                    file_size_mb = file_path_obj.stat().st_size / (1024 * 1024)
                    is_substantial = file_size_mb > 0.5  # Larger than 500KB
                except:
                    is_substantial = True  # Include if we can't check size
                
                if is_shallow or is_audio_folder or is_substantial:
                    filtered_stragglers.append(file_path)
            
            print(f"  📄 Found {original_count} total, filtered to {len(filtered_stragglers)} relevant files")
            stragglers = filtered_stragglers
        else:
            print(f"  📄 Found {len(stragglers)} audio files")
        
        all_stragglers.extend(stragglers)
        
        # Show examples
        for example in stragglers[:3]:
            print(f"    📀 {Path(example).name}")
        if len(stragglers) > 3:
            print(f"    ... and {len(stragglers) - 3} more")
    
    return all_stragglers

# Find the stragglers
stragglers = find_audio_stragglers_smart(STRAGGLER_LOCATIONS)
print(f"\n📊 TOTAL STRAGGLERS FOUND: {len(stragglers)}")

# ================================================================
# STEP 2: Organize Stragglers (DRY RUN)
# ================================================================

if stragglers:
    print(f"\n🔄 ORGANIZING {len(stragglers)} STRAGGLERS (DRY RUN)")
    print("="*50)
    print("This shows what would happen without moving files")
    
    # Process with low interaction (ADHD-friendly)
    results = organizer.interactive_batch_process(
        stragglers, 
        confidence_threshold=0.5,  # Lower threshold = fewer interruptions
        dry_run=True
    )
    
    print(f"\n📋 DRY RUN COMPLETE!")
    successful_results = [r for r in results if r]
    print(f"✅ Would organize: {len(successful_results)} files")
    print(f"⏭️ Would skip: {len(stragglers) - len(successful_results)} files")
    
    # Show some examples of what would happen
    print(f"\n🎯 SAMPLE ORGANIZATION RESULTS:")
    for i, result in enumerate(successful_results[:5], 1):
        print(f"{i}. {result.get('suggested_filename', 'Unknown')}")
        print(f"   📂 {result.get('category', 'unknown')} → {result.get('mood', 'unknown')}")
        print(f"   ⚡ Energy: {result.get('energy_level', 0)}/10")
    
    if len(successful_results) > 5:
        print(f"   ... and {len(successful_results) - 5} more files")
    
    print(f"\n💡 To actually move files:")
    print(f"   results = organizer.interactive_batch_process(stragglers, dry_run=False)")

else:
    print("🎉 No stragglers found! Your audio is already organized.")

# ================================================================
# STEP 3: Attention Episode Cue Suggestions
# ================================================================

print(f"\n" + "="*60)
print("🎬 ATTENTION IS ALL YOU NEED - CUE SUGGESTIONS")
print("="*60)

def suggest_cues_for_attention_scene(scene_description, max_suggestions=5):
    """Get cue suggestions specifically tuned for Attention episode"""
    print(f"🎬 Scene: {scene_description}")
    print(f"🔍 Searching your library of {len(organizer.learning_data['classifications'])} analyzed files...")
    
    matches = []
    scene_words = scene_description.lower().split()
    
    # Search through your organized library
    for classification_entry in organizer.learning_data['classifications']:
        classification = classification_entry.get('classification', {})
        filename = classification_entry.get('filename', '')
        
        relevance = 0
        
        # Mood matching
        mood = classification.get('mood', '').lower()
        for word in scene_words:
            if word in mood:
                relevance += 3
        
        # Tags matching
        tags = classification.get('tags', [])
        for tag in tags:
            tag_lower = tag.lower()
            for word in scene_words:
                if word in tag_lower:
                    relevance += 2
        
        # Reasoning/description matching
        reasoning = classification.get('reasoning', '').lower()
        for word in scene_words:
            if word in reasoning:
                relevance += 1
        
        # Special attention/transformer theme boosts
        attention_keywords = ['attention', 'focus', 'transform', 'pattern', 'neural', 'process', 
                             'mechanism', 'layer', 'head', 'query', 'key', 'value', 'matrix']
        
        file_text = f"{filename} {mood} {' '.join(tags)} {reasoning}".lower()
        
        for keyword in attention_keywords:
            if keyword in scene_description.lower() and keyword in file_text:
                relevance += 4
        
        # Digital/AI consciousness themes
        consciousness_keywords = ['digital', 'consciousness', 'awakening', 'emergence', 'memory', 
                                 'synthetic', 'artificial', 'machine', 'electronic']
        
        for keyword in consciousness_keywords:
            if keyword in scene_description.lower() and keyword in file_text:
                relevance += 3
        
        if relevance > 0:
            matches.append({
                'filename': filename,
                'relevance': relevance,
                'mood': classification.get('mood', 'unknown'),
                'energy': classification.get('energy_level', 0),
                'intensity': classification.get('intensity', 'unknown'),
                'tags': classification.get('tags', []),
                'reasoning': classification.get('reasoning', ''),
                'confidence': classification.get('confidence', 0),
                'category': classification.get('category', 'unknown'),
                'bpm': classification.get('bpm', 0)
            })
    
    # Sort by relevance
    matches.sort(key=lambda x: x['relevance'], reverse=True)
    
    # Display results
    if matches:
        print(f"✨ Found {len(matches)} potential matches:")
        print()
        
        for i, match in enumerate(matches[:max_suggestions], 1):
            print(f"{i}. 🎵 {match['filename']}")
            print(f"   🎭 {match['mood']} | ⚡ Energy: {match['energy']}/10 | 🎯 {match['intensity']}")
            if match['bpm'] > 0:
                print(f"   🥁 {match['bpm']} BPM | 📂 {match['category']}")
            else:
                print(f"   📂 {match['category']}")
            print(f"   🏷️ Tags: {', '.join(match['tags'][:4])}")
            print(f"   💭 Why: {match['reasoning'][:120]}...")
            print(f"   📊 Relevance: {match['relevance']} | 🔥 Confidence: {match['confidence']:.1%}")
            print()
    else:
        print("🤔 No matches found for that description.")
        print("💡 Try describing differently or check if more audio needs organizing!")
    
    return matches[:max_suggestions]

# Ready to use cue suggestion function
print("🎯 READY FOR CUE SUGGESTIONS!")
print("Try these commands:")
print()
print('suggest_cues_for_attention_scene("quiet moment of attention mechanism awakening")')
print('suggest_cues_for_attention_scene("intense transformation of neural attention patterns")')
print('suggest_cues_for_attention_scene("breakthrough moment of pattern recognition")')
print('suggest_cues_for_attention_scene("human discovering transformer architecture")')
print('suggest_cues_for_attention_scene("digital consciousness processing information")')
print()
print("🚀 System ready! What scene do you need audio for?")
suggest_cues_for_attention_scene("quiet moment of attention mechanism awakening")


# Fix the string/Path issue in batch processing
def interactive_batch_process_fixed(self, file_list, confidence_threshold=0.7, dry_run=True):
    """Fixed batch processing that handles string paths properly."""
    if not file_list:
        print("❌ No files provided for processing")
        return []
    
    # Set confidence threshold for this batch
    original_threshold = self.interaction_threshold
    self.interaction_threshold = confidence_threshold
    
    print(f"\n🎵 Starting interactive batch processing...")
    print(f"📁 Files to process: {len(file_list)}")
    print(f"🎯 Confidence threshold: {confidence_threshold:.1%}")
    print(f"🔄 Dry run: {dry_run}")
    print("=" * 50)
    
    results = []
    processed_count = 0
    skipped_count = 0
    
    try:
        for i, file_path in enumerate(file_list):
            # Convert to Path object if it's a string
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            try:
                print(f"\n[{i+1}/{len(file_list)}] Processing: {file_path.name}")
                
                # Get classification
                classification = self.classify_audio_file(file_path)
                
                if classification:
                    target_folder = self.determine_target_folder(classification)
                    suggested_filename = classification.get('suggested_filename', '')
                    
                    print(f"  📂 Category: {classification.get('category', 'unknown')}")
                    print(f"  🎭 Mood: {classification.get('mood', 'unknown')}")
                    print(f"  ⚡ Intensity: {classification.get('intensity', 'unknown')}")
                    print(f"  🔥 Energy: {classification.get('energy_level', 0)}/10")
                    print(f"  🎯 Confidence: {classification.get('confidence', 0):.1%}")
                    print(f"  📍 Target: {target_folder}")
                    
                    if not dry_run:
                        new_path = self.move_file_safely(file_path, target_folder, suggested_filename)
                        if new_path:
                            print(f"  ✅ Moved to: {new_path}")
                            classification['final_path'] = str(new_path)
                        else:
                            print(f"  ❌ Failed to move")
                    else:
                        print(f"  🔄 [DRY RUN] Would move to: {target_folder}")
                    
                    results.append(classification)
                    processed_count += 1
                else:
                    print(f"  ❌ Classification failed")
                    skipped_count += 1
                
                # Small delay to avoid hitting API limits
                import time
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                skipped_count += 1
    
    finally:
        # Restore original threshold
        self.interaction_threshold = original_threshold
    
    print(f"\n{'='*50}")
    print(f"🎉 Batch processing completed!")
    print(f"✅ Processed: {processed_count}")
    print(f"⏭️ Skipped: {skipped_count}")
    print(f"📊 Total: {len(file_list)}")
    print(f"📈 Success rate: {processed_count/len(file_list):.1%}")
    
    return results

# Apply the fix
import types
organizer.interactive_batch_process = types.MethodType(interactive_batch_process_fixed, organizer)

print("🔧 Fixed the batch processing string/Path issue!")

# Try a small test first
test_batch = stragglers[:3]
print(f"\n🧪 Testing with 3 files first...")
results = organizer.interactive_batch_process(test_batch, dry_run=True)

if len(results) > 0:
    print(f"✅ SUCCESS! {len(results)} files processed")
    print(f"🚀 Ready to process all 674 files!")
    print(f"results = organizer.interactive_batch_process(stragglers, dry_run=True)")
else:
    print(f"❌ Still failing - need more debugging")

# Main execution example
if __name__ == "__main__":
    # Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    BASE_DIRECTORY = os.getenv('AUDIOAI_BASE_DIRECTORY')
    if not BASE_DIRECTORY:
        print("❌ Please set AUDIOAI_BASE_DIRECTORY environment variable")
        print("   export AUDIOAI_BASE_DIRECTORY='/path/to/your/audio/library'")
        exit(1)    
    # Directories to scan from environment
    from config import DIRECTORIES_TO_SCAN
    if not DIRECTORIES_TO_SCAN:
        DIRECTORIES_TO_SCAN = [BASE_DIRECTORY]
    
    # Initialize organizer
    organizer = AdaptiveAudioOrganizer(OPENAI_API_KEY, BASE_DIRECTORY)
    
    # Show current learning state
    organizer.show_learning_stats()
    
    # Example: Process a single file first
    # result = organizer.process_file_interactive("test_audio.mp3", dry_run=True)
    
    # Example: Run full system sweep
    # results = organizer.sweep_system(DIRECTORIES_TO_SCAN, dry_run=True)
    
    print("\nAudioAI Organizer ready!")
    print("Use organizer.process_file() to classify individual files")
    print("Use organizer.sweep_system() to process entire directories")

