from __future__ import annotations
from typing import Dict, Any, List
from transformers import pipeline
from utils.config import Config
from utils.helpers import write_json
from pathlib import Path


# Load emotion model only once for efficiency
print("[🧠] Loading emotion detection model...")
_EMO = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
)
print("✅ Emotion model loaded successfully.\n")


def add_emotion_to_segments(gloss_json: Dict[str, Any], cfg: Config = None) -> Dict[str, Any]:
    """
    Add emotion labels and confidence scores to each segment in gloss_json.
    
    The emotion analysis helps generate more expressive sign language animations:
    - Happy/excited → animated signing
    - Sad/angry → slower, more deliberate signing  
    - Neutral → standard signing
    
    Saves output to 'output' directory.
    """
    if cfg is None:
        cfg = Config()
    cfg.ensure_dirs()

    print("[🎭] Analyzing emotions for each text segment...")
    out_segments: List[Dict[str, Any]] = []

    for i, seg in enumerate(gloss_json.get("segments", []), 1):
        text = seg.get("text", "").strip()
        if not text:
            seg_out = dict(seg)
            seg_out["emotion"] = {"label": "neutral", "score": 0.0}
            out_segments.append(seg_out)
            continue

        # Get model predictions
        scores = _EMO(text)[0]  # list of {label, score}
        top = max(scores, key=lambda x: x["score"]) if scores else {"label": "neutral", "score": 0.0}

        seg_out = dict(seg)
        seg_out["emotion"] = {"label": top["label"], "score": float(top["score"])}
        out_segments.append(seg_out)

        print(f"  [{i}] '{text[:40]}...' → {top['label']} ({top['score']:.2f})")

    out = {"segments": out_segments}

    # Save the updated emotion-enriched JSON
    output_path = Path(cfg.output_dir) / "emotion_output.json"
    write_json(output_path, out)
    print(f"\n✅ Emotion analysis complete. Saved: {output_path.name}")

    return out
