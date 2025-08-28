# stage1_youtube.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import subprocess, shutil
from yt_dlp import YoutubeDL
from config import Config

def _ffmpeg_bin() -> str:
    # Try PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common Windows install path (change if yours is different)
    candidate = r"C:\ffmpeg\ffmpeg-2025-08-25-git-1b62f9d3ae-full_build\bin\ffmpeg.exe"
    if Path(candidate).exists():
        return candidate
    raise RuntimeError(
        "FFmpeg not found. Ensure it's on PATH or set the path in _ffmpeg_bin()."
    )

def _run_ffmpeg(args: list[str]) -> None:
    ffbin = _ffmpeg_bin()
    proc = subprocess.run([ffbin, "-y", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

def download_audio_to_wav(url: str, cfg: Optional[Config] = None) -> Path:
    cfg = cfg or Config()
    cfg.ensure_dirs()

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

    safe_name = downloaded_path.stem
    wav_path = cfg.output_dir / f"{safe_name}.wav"
    ff_args = ["-i", str(downloaded_path), "-ac", "1", "-ar", str(cfg.sample_rate), str(wav_path)]
    _run_ffmpeg(ff_args)
    return wav_path
