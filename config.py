import os
from pathlib import Path
from typing import List

def _require_env(var: str, hint: str = "") -> str:
    val = os.getenv(var, "").strip()
    if not val:
        msg = f"Missing required environment variable: {var}"
        if hint:
            msg += f"\nHint: {hint}"
        raise RuntimeError(msg)
    return val

def _optional_env_path(var: str, default: str = "") -> Path:
    val = os.getenv(var, default).strip()
    return Path(val) if val else Path()

def _env_paths_list(var: str) -> List[Path]:
    # Comma-separated list of absolute paths
    raw = os.getenv(var, "").strip()
    if not raw:
        return []
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]

# Required
OPENAI_API_KEY = _require_env(
    "OPENAI_API_KEY",
    "export OPENAI_API_KEY='your-api-key-here'"
)

BASE_DIRECTORY = Path(_require_env(
    "AUDIOAI_BASE_DIRECTORY",
    "export AUDIOAI_BASE_DIRECTORY='/path/to/your/audio/library'"
))

# Optional lists/flags
DIRECTORIES_TO_SCAN = _env_paths_list("AUDIOAI_SCAN_DIRS")  # optional override
TEST_AUDIO_FILE = os.getenv("AUDIOAI_TEST_FILE", "").strip()  # optional single test file
DRY_RUN_DEFAULT = os.getenv("AUDIOAI_DRY_RUN", "true").lower() == "true"


def resolve_test_file() -> Path:
    return Path(TEST_AUDIO_FILE) if TEST_AUDIO_FILE else Path()


def ensure_base_structure() -> Path:
    # Use centralized local metadata root (Rule #3)
    from gdrive_integration import get_metadata_root
    meta_dir = get_metadata_root()
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir