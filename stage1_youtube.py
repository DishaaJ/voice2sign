from __future__ import annotations
from pathlib import Path
from typing import Optional
import subprocess, shutil
from yt_dlp import YoutubeDL
from config import Config


def _ffmpeg_bin() -> str:
    """
    Locate ffmpeg binary. 
    Relies on ffmpeg being available on PATH.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "FFmpeg not found. Please install it and add to PATH (e.g., C:\\ffmpeg\\bin on Windows)."
    )


def _run_ffmpeg(args: list[str]) -> None:
    ffbin = _ffmpeg_bin()
    proc = subprocess.run([ffbin, "-y", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))


def download_audio_to_wav(url: str, cfg: Optional[Config] = None) -> Path:
    """
    Download audio from a YouTube URL and convert it to 16kHz mono WAV.
    """
    cfg = cfg or Config()
    cfg.ensure_dirs()

    # Step 1: Download best available audio
    ydl_out = cfg.tmp_dir / "%(title)s.%(ext)s"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(ydl_out),
        "quiet": True,
        "noprogress": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded_path = Path(ydl.prepare_filename(info))

    # Step 2: Convert to 16kHz mono WAV using ffmpeg
    safe_name = downloaded_path.stem
    wav_path = cfg.output_dir / f"{safe_name}.wav"
    ff_args = ["-i", str(downloaded_path), "-ac", "1", "-ar", str(cfg.sample_rate), str(wav_path)]
    _run_ffmpeg(ff_args)

    return wav_path
