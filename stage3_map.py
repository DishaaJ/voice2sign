from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import csv
from config import Config
from utils import write_json


def load_sign_dict(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row.get("gloss", "").strip().upper()
            file = row.get("file", "").strip()
            if gloss and file:
                mapping[gloss] = file
    return mapping


ALPHABET = {c: f"FINGERSPELL_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}


def token_to_assets(token: str, mapping: dict, cfg: Config) -> List[Dict[str, str]]:
    """Return a list of assets for a token. Typically length 1; if missing, fall back to fingerspelling."""
    if token in {"|"}:
        return [{"type": "pause", "dur": 0.3}]  # 300 ms pause
    if token in mapping:
        path = cfg.signs_dir / mapping[token]
        return [{"type": "video", "path": str(path)}]
    # Fallback: fingerspelling (one asset per letter)
    assets: List[Dict[str, str]] = []
    for ch in token:
        if ch in ALPHABET:
            assets.append({"type": "fingerspell", "label": ALPHABET[ch]})
    if not assets:
        assets.append({"type": "text", "label": token})
    return assets


def build_sign_timeline(gloss_json: Dict[str, Any], cfg: Config) -> Dict[str, Any]:
    mapping = load_sign_dict(cfg.sign_dict_csv)
    timeline = []
    for seg in gloss_json["segments"]:
        seg_items = []
        for tok in seg["gloss"]:
            seg_items.extend(token_to_assets(tok, mapping, cfg))
        timeline.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "gloss": seg["gloss"],
            "emotion": seg.get("emotion"),
            "items": seg_items,
        })
    return {"timeline": timeline}


def save_timeline_json(timeline: Dict[str, Any], wav_stem: str, cfg: Config) -> Path:
    path = cfg.output_dir / f"{wav_stem}_sign_timeline.json"
    write_json(path, timeline)
    print(f"Saved: {path.name}")
    return path
