from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import os
from utils.config import Config
from utils.helpers import write_json

# ==========================================================
# STEP 1: Detect dataset automatically
# ==========================================================
dataset_path = Path(os.getcwd()) / "datasets" / "data"

if dataset_path.exists():
    print("✅ [Dataset] Found ISL dataset at:", dataset_path)
    print("📂 Sample contents:", os.listdir(dataset_path)[:5], "\n")
else:
    print("⚠️  [Dataset] Folder NOT found at 'datasets/data'. This is optional - will use fallback methods.\n")


# ==========================================================
# STEP 2: Load all sign mappings from dataset
# ==========================================================
def load_sign_dict(dataset_path: Path) -> dict:
    """Auto-load ISL sign mappings directly from dataset folder."""
    if not dataset_path.exists():
        print("⚠️  Dataset path not found - using text fallback only")
        return {}

    mapping: dict[str, str] = {}
    for folder in dataset_path.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.suffix.lower() in [".jpg", ".png", ".mp4", ".gif"]:
                    mapping[folder.name.upper()] = str(file)
                    break  # One representative file per sign/letter

    print(f"✅ Loaded {len(mapping)} ISL signs from dataset.")
    return mapping


# ==========================================================
# STEP 3: Define fallback alphabet mapping
# ==========================================================
ALPHABET = {c: f"FINGERSPELL_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


# ==========================================================
# STEP 4: Convert recognized tokens to ISL gesture/video assets
# ==========================================================
def token_to_assets(token: str, mapping: dict, cfg: Config) -> List[Dict[str, str]]:
    """Return list of gesture/image/video assets for a token."""
    token = token.upper()

    # Pause marker
    if token == "|":
        return [{"type": "pause", "dur": 0.4}]

    # Exact ISL match
    if token in mapping:
        path = Path(mapping[token])
        return [{"type": "image", "path": str(path), "label": token}]

    # Fallback: fingerspelling each character
    assets: List[Dict[str, str]] = []
    for ch in token:
        if ch in ALPHABET:
            assets.append({"type": "fingerspell", "label": ALPHABET[ch], "char": ch})

    # If still no match → text fallback
    if not assets:
        assets.append({"type": "text", "label": token})

    return assets


# ==========================================================
# STEP 5: Build timeline combining gloss + emotion
# ==========================================================
def build_sign_timeline(gloss_json: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    """Create a full ISL sign sequence timeline."""
    print("[🧩] Building ISL sign timeline...")
    mapping = load_sign_dict(dataset_path)
    timeline = []

    for seg in gloss_json.get("segments", []):
        seg_items = []
        for tok in seg.get("gloss", []):
            seg_items.extend(token_to_assets(tok, mapping, cfg))
        
        timeline.append({
            "id": seg.get("id"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text"),
            "gloss": seg.get("gloss"),
            "emotion": seg.get("emotion", {}),
            "items": seg_items,
        })

        print(f"  ▶ Segment {seg['id']}: {seg['text'][:40]}... → {len(seg_items)} items")

    print("\n✅ Sign timeline built successfully.")
    return {"timeline": timeline}


# ==========================================================
# STEP 6: Save generated timeline JSON
# ==========================================================
def save_timeline_json(timeline: Dict[str, Any], wav_stem: str, cfg: Config) -> Path:
    """Save sign timeline to output folder."""
    cfg.ensure_dirs()
    path = cfg.output_dir / f"{wav_stem}_sign_timeline.json"
    write_json(path, timeline)
    print(f"💾 Saved timeline JSON → {path.name}")
    return path


# ==========================================================
# STEP 7: Optional standalone test
# ==========================================================
if __name__ == "__main__":
    cfg = Config()
    mapping = load_sign_dict(dataset_path)
    print(f"\nPreview of loaded ISL mappings: {list(mapping.keys())[:10]}")
