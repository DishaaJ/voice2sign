from pathlib import Path
from config import Config
from utils import read_json
from stage2_nlp import process_segments_to_gloss, save_gloss_json
from stage2_emotion import add_emotion_to_segments

# ==========================================================
# INITIAL SETUP
# ==========================================================
cfg = Config()
cfg.ensure_dirs()

# ⚠️ Replace this with your video name (no file extension)
# It must match the WAV/transcript name created in Stage 1
wav_stem = "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)"

print("\n🎬 [Stage 2 Test] Text → Gloss + Emotion Mapping")
print("--------------------------------------------------")

# ==========================================================
# STEP 1: LOAD TRANSCRIPT JSON
# ==========================================================
transcript_path = cfg.output_dir / f"{wav_stem}_transcript.json"

if not transcript_path.exists():
    print(f"❌ Transcript not found: {transcript_path}")
    print("➡️ Run test_stage1.py first to generate the transcript.")
    exit(1)

print(f"📂 Loading transcript from: {transcript_path}")
transcript = read_json(transcript_path)
print(f"✅ Loaded transcript with {len(transcript.get('segments', []))} segments.\n")

# ==========================================================
# STEP 2: CONVERT TO GLOSS TOKENS
# ==========================================================
try:
    print("🔠 Converting English text to Sign-Language Gloss...")
    gloss_json = process_segments_to_gloss(transcript["segments"], cfg)
    print(f"✅ Gloss tokens generated for {len(gloss_json['segments'])} segments.\n")
except Exception as e:
    print(f"❌ Error while generating gloss tokens:\n{e}")
    exit(1)

# ==========================================================
# STEP 3: ADD EMOTION LABELS
# ==========================================================
try:
    print("🎭 Adding emotion labels (using DistilBERT)...")
    gloss_with_emotion = add_emotion_to_segments(gloss_json)
    print("✅ Emotion labels added successfully.\n")
except Exception as e:
    print(f"❌ Error during emotion labeling:\n{e}")
    exit(1)

# ==========================================================
# STEP 4: SAVE OUTPUT JSON
# ==========================================================
try:
    gloss_path = save_gloss_json(gloss_with_emotion, wav_stem, cfg)
    print(f"💾 Gloss + Emotion JSON saved successfully → {gloss_path}")
except Exception as e:
    print(f"❌ Error saving output JSON:\n{e}")
    exit(1)

# ==========================================================
# STEP 5: PREVIEW
# ==========================================================
print("\n📘 === Preview of Processed Segments ===")
for i, seg in enumerate(gloss_with_emotion["segments"][:3], start=1):
    gloss = " ".join(seg["gloss"])
    emo = seg.get("emotion", {}).get("label", "neutral")
    print(f" {i}. {seg['text']} → [{gloss}] ({emo})")

print("\n✅ Stage 2 completed successfully!")
