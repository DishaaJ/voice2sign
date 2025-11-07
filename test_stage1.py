from pathlib import Path
from config import Config
from stage1_youtube import download_audio_to_wav
from stage1_transcribe import transcribe_wav

cfg = Config()
cfg.ensure_dirs()

# Replace with a short YouTube video URL you want to test
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

print("[TEST] Downloading & converting audio...")
wav_path = download_audio_to_wav(url, cfg)

print("[TEST] Transcribing audio...")
result = transcribe_wav(wav_path, cfg)

print("\n=== Transcript Preview ===")
print(result["text"][:300])  # show first 300 chars
