from __future__ import annotations
import json
from pathlib import Path
from typing import Any




def safe_stem(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)




def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)




def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)