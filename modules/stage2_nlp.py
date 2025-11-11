from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import spacy
from utils.config import Config
from utils.helpers import write_json

# -------------------- Load spaCy Model --------------------
try:
    print("[🧩] Loading spaCy model (en_core_web_sm)...")
    _NLP = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully.\n")
except OSError:
    raise SystemExit("❌ spaCy model not found. Run this first:\n   python -m spacy download en_core_web_sm")


# -------------------- Stopword Customization --------------------
BASE_STOP = {
    "um", "uh", "like", "you_know", "i_mean", "basically", "literally",
}

SPACY_STOP = set(w for w in _NLP.Defaults.stop_words)
for keep in ("no", "not", "never"):
    SPACY_STOP.discard(keep)


# -------------------- Text → Gloss Conversion --------------------
def text_to_gloss_tokens(text: str, keep_negation: bool = True) -> List[str]:
    """Convert English sentence into sign-language style gloss tokens."""
    doc = _NLP(text)
    toks: List[str] = []

    for t in doc:
        if t.is_space:
            continue

        lower = t.text.lower()
        if lower in BASE_STOP:
            continue

        if t.is_punct:
            if t.text in {".", "!", "?", ";"}:
                toks.append("|")  # pause marker
            continue

        lemma = t.lemma_.lower().strip()
        if not lemma:
            continue
        if lemma in SPACY_STOP:
            continue

        if lemma.isnumeric():
            toks.append(lemma.upper())
            continue

        if keep_negation and lemma in {"no", "not", "never"}:
            toks.append(lemma.upper())
            continue

        if t.pos_ in {"PROPN", "NOUN", "VERB", "ADJ", "ADV", "AUX", "PRON"}:
            toks.append(lemma.upper())

    # Clean redundant pauses
    cleaned: List[str] = []
    for tok in toks:
        if tok == "|" and (not cleaned or cleaned[-1] == "|"):
            continue
        cleaned.append(tok)

    return cleaned


# -------------------- Segment Processing --------------------
def process_segments_to_gloss(segments: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    """Convert each transcript segment into gloss tokens."""
    print("[🔠] Converting transcript text → sign-language gloss form...")
    out_segments = []

    for i, seg in enumerate(segments, 1):
        text = seg.get("text", "").strip()
        if not text:
            print(f"  [{i}] Empty segment skipped.")
            continue

        tokens = text_to_gloss_tokens(text, keep_negation=cfg.keep_negation)
        out_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "gloss": tokens,
        })

        preview = " ".join(tokens[:8]) + ("..." if len(tokens) > 8 else "")
        print(f"  [{i}] '{text[:40]}...' → {preview}")

    out = {"segments": out_segments}
    print("\n✅ Gloss tokenization complete.")
    return out


# -------------------- Save Gloss JSON --------------------
def save_gloss_json(gloss_data: Dict[str, Any], wav_stem: str, cfg: Config) -> Path:
    """Save gloss tokens JSON for later use in Stage 3 mapping."""
    path = cfg.output_dir / f"{wav_stem}_gloss.json"
    write_json(path, gloss_data)
    print(f"💾 Saved gloss data: {path.name}")
    return path
