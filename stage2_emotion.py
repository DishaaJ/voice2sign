from __future__ import annotations
from typing import Dict, Any, List
from transformers import pipeline
from config import Config

# Load pipeline once
_EMO = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    return_all_scores=True,
)


def add_emotion_to_segments(gloss_json: Dict[str, Any]) -> Dict[str, Any]:
    out_segments: List[Dict[str, Any]] = []
    for seg in gloss_json["segments"]:
        text = seg["text"]
        scores = _EMO(text)[0]  # list of {label, score}
        top = max(scores, key=lambda x: x["score"]) if scores else {"label": "neutral", "score": 0.0}
        seg_out = dict(seg)
        seg_out["emotion"] = {"label": top["label"], "score": float(top["score"])}
        out_segments.append(seg_out)
    return {"segments": out_segments}
