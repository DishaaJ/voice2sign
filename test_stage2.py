from pathlib import Path
from config import Config
from utils import read_json
from stage2_nlp import process_segments_to_gloss, save_gloss_json
from stage2_emotion import add_emotion_to_segments

cfg = Config()
cfg.ensure_dirs()

# Replace this with your actual video file name (without extension)
wav_stem = "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)"

# 1) Load transcript JSON
transcript_path = cfg.output_dir / f"{wav_stem}_transcript.json"
print(f"[Stage 2] Loading transcript: {transcript_path}")
transcript = read_json(transcript_path)

# 2) Convert to gloss tokens
gloss_json = process_segments_to_gloss(transcript["segments"], cfg)

# 3) Add emotion labels
gloss_with_emotion = add_emotion_to_segments(gloss_json)

# 4) Save output
gloss_path = save_gloss_json(gloss_with_emotion, wav_stem, cfg)

print(f"[Stage 2] Gloss + Emotion JSON saved at: {gloss_path}")
