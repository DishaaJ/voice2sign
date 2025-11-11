from __future__ import annotations
from pathlib import Path
from typing import Optional
import subprocess
import shutil
from yt_dlp import YoutubeDL
from utils.config import Config


def _ffmpeg_bin() -> str:
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


def download_youtube_audio(url: str, cfg: Optional[Config] = None) -> Path:
    """
    Download audio from a YouTube URL and convert it to 16kHz mono WAV.
    Returns: Path to the created WAV file.
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

    ff_args = [
        "-i", str(downloaded_path),
        "-ac", "1",  # mono
        "-ar", str(cfg.sample_rate),  # sample rate (16k typical)
        str(wav_path)
    ]
    _run_ffmpeg(ff_args)

    return wav_path
