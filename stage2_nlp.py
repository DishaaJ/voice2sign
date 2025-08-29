from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import spacy
from config import Config
from utils import write_json

# Load once at import time (small model is fine for this stage)
try:
    _NLP = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit("spaCy model not found. Run: python -m spacy download en_core_web_sm")

# Keep negation; drop typical fillers
BASE_STOP = {
    "um", "uh", "like", "you_know", "i_mean", "basically", "literally",
}

# spaCy stop words (we remove NO/NOT/NEVER from it)
SPACY_STOP = set(w for w in _NLP.Defaults.stop_words)
for keep in ("no", "not", "never"):
    if keep in SPACY_STOP:
        SPACY_STOP.remove(keep)


def text_to_gloss_tokens(text: str, keep_negation: bool = True) -> List[str]:
    doc = _NLP(text)
    toks: List[str] = []
    for t in doc:
        if t.is_space:
            continue
        lower = t.text.lower()
        if lower in BASE_STOP:
            continue
        if t.is_punct:
            # turn hard punctuation into pause marker
            if t.text in {".", "!", "?", ";"}:
                toks.append("|")
            continue
        lemma = t.lemma_.lower().strip()
        if not lemma:
            continue
        if lemma in SPACY_STOP:
            continue
        # Skip pure digits but keep numbers as-is (optional: convert to NUM)
        if lemma.isnumeric():
            toks.append(lemma.upper())
            continue
        # Keep negation words explicitly
        if keep_negation and lemma in {"no", "not", "never"}:
            toks.append(lemma.upper())
            continue
        # Proper nouns → upper
        if t.pos_ in {"PROPN", "NOUN", "VERB", "ADJ", "ADV", "AUX", "PRON"}:
            toks.append(lemma.upper())
    # Collapse duplicates and excess pauses
    cleaned: List[str] = []
    for tok in toks:
        if tok == "|" and (not cleaned or cleaned[-1] == "|"):
            continue
        cleaned.append(tok)
    return cleaned


def process_segments_to_gloss(segments: List[Dict[str, Any]], cfg: Config) -> Dict[str, Any]:
    out_segments = []
    for seg in segments:
        tokens = text_to_gloss_tokens(seg["text"], keep_negation=cfg.keep_negation)
        out_segments.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "gloss": tokens,
        })
    return {"segments": out_segments}


def save_gloss_json(gloss_data: Dict[str, Any], wav_stem: str, cfg: Config) -> Path:
    path = cfg.output_dir / f"{wav_stem}_gloss.json"
    write_json(path, gloss_data)
    print(f"Saved: {path.name}")
    return path
