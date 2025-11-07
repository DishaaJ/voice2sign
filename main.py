from __future__ import annotations
from pathlib import Path
from config import Config
from utils import read_json
from stage1_youtube import download_youtube_audio
from stage1_transcribe import transcribe_wav
from stage2_nlp import process_segments_to_gloss, save_gloss_json
from stage2_emotion import add_emotion_to_segments
from stage3_map import build_sign_timeline, save_timeline_json


def main():
    # ================================
    # STEP 1: INITIALIZE CONFIGURATION
    # ================================
    cfg = Config()
    cfg.ensure_dirs()

    # Example YouTube video (replace with your own)
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    print("\n🎬 STEP 1: Downloading & converting YouTube audio...")
    wav_path = download_youtube_audio(url, cfg)

    print("\n🧠 STEP 2: Transcribing speech to text...")
    transcript_json = transcribe_wav(wav_path, cfg)

    print("\n📘 STEP 3: Generating gloss tokens from transcript...")
    gloss_json = process_segments_to_gloss(transcript_json["segments"], cfg)

    print("\n🎭 STEP 4: Adding emotion analysis to each segment...")
    gloss_with_emotion = add_emotion_to_segments(gloss_json, cfg)
    wav_stem = wav_path.stem
    gloss_path = save_gloss_json(gloss_with_emotion, wav_stem, cfg)

    print("\n🤟 STEP 5: Mapping gloss to ISL dataset...")
    timeline = build_sign_timeline(gloss_with_emotion, cfg)
    save_timeline_json(timeline, wav_stem, cfg)

    print("\n✅ Pipeline completed successfully!")
    print(f"🔊 Audio file: {wav_path}")
    print(f"📜 Transcript JSON: {wav_stem}_transcript.json")
    print(f"🧩 Gloss JSON: {wav_stem}_gloss.json")
    print(f"🎭 Emotion-enhanced JSON: {wav_stem}_gloss.json")
    print(f"🤟 ISL Timeline JSON: {wav_stem}_sign_timeline.json")


if __name__ == "__main__":
    main()
