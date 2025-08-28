from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import whisper
from tqdm import tqdm
from config import Config
from utils import write_json




def transcribe_wav(wav_path: Path, cfg: Optional[Config] = None) -> Dict[str, Any]:
    cfg = cfg or Config()
    cfg.ensure_dirs()


    print(f"[Whisper] Loading model: {cfg.whisper_model}")
    model = whisper.load_model(cfg.whisper_model)


    print(f"[Whisper] Transcribing: {wav_path.name}")
    result = model.transcribe(str(wav_path), language=cfg.language, verbose=False)


# Normalize output structure
    segments = []
    for seg in result.get("segments", []):
        segments.append({
        "id": seg.get("id"),
        "start": round(float(seg.get("start", 0.0)), 3),
        "end": round(float(seg.get("end", 0.0)), 3),
        "text": seg.get("text", "").strip(),
    })


    out = {
        "audio": str(wav_path),
        "language": result.get("language"),
        "text": result.get("text", "").strip(),
        "segments": segments,
    }


# Save rich JSON and a flat TXT
    json_path = Path(cfg.output_dir) / f"{wav_path.stem}_transcript.json"
    txt_path = Path(cfg.output_dir) / f"{wav_path.stem}_transcript.txt"


    write_json(json_path, out)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(out["text"] + "\n")


    print(f"Saved: {json_path.name} and {txt_path.name}")
    return out