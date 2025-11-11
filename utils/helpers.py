from __future__ import annotations
import json
from pathlib import Path
from typing import Any




def safe_stem(s: str) -> str:
    """Sanitize filename by replacing special characters."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)




def write_json(path: Path, data: Any) -> None:
    """Write data to JSON file with proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)




def read_json(path: Path) -> Any:
    """Read JSON file and return parsed data."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
