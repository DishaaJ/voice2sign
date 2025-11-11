from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    whisper_model: str = "base"
    language: str | None = None

    root: Path = Path(__file__).parent.parent.resolve()
    assets_dir: Path = root / "assets"
    signs_dir: Path = assets_dir / "signs"
    sign_dict_csv: Path = assets_dir / "sign_dictionary.csv"
    output_dir: Path = root / "output"
    tmp_dir: Path = output_dir / "tmp"

    sample_rate: int = 16000
    mono: bool = True

    keep_negation: bool = True

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.signs_dir.mkdir(parents=True, exist_ok=True)
