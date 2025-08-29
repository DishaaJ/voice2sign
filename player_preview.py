from __future__ import annotations
import time
from pathlib import Path
import cv2
from utils import read_json


def _play_video(path: Path) -> None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Sign Preview", frame)
        if cv2.waitKey(25) & 0xFF == 27:  # ESC to exit early
            break
    cap.release()


def preview_timeline(json_path: Path) -> None:
    data = read_json(json_path)
    cv2.namedWindow("Sign Preview", cv2.WINDOW_NORMAL)
    for seg in data.get("timeline", []):
        for item in seg.get("items", []):
            if item["type"] == "video":
                p = Path(item["path"])
                if p.exists():
                    _play_video(p)
                else:
                    # fallback to text if file missing
                    txt = p.name
                    img = 255 * (1 - cv2.putText(
                        255 * np.ones((400, 800, 3), dtype="uint8"),
                        txt[:60], (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2
                    ))
                    cv2.imshow("Sign Preview", img)
                    cv2.waitKey(500)
            elif item["type"] == "pause":
                time.sleep(float(item.get("dur", 0.3)))
            else:
                # fingerspell or text label
                label = item.get("label", "?")
                img = 255 * np.ones((400, 800, 3), dtype="uint8")
                cv2.putText(img, label[:60], (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                cv2.imshow("Sign Preview", img)
                cv2.waitKey(500)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    import numpy as np  # local import to avoid top-level dep

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to *_sign_timeline.json")
    args = parser.parse_args()
    preview_timeline(Path(args.json))
