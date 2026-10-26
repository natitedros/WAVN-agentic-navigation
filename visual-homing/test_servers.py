"""
Checks both servers together: Grounding DINO finds objects, BLIP-2 describes each one.

Usage:
  export BLIP2_URL=https://....trycloudflare.com
  export BLIP2_TOKEN=....
  python test_servers.py path/to/photo.jpg "a door, a car, a tree, a sign"
"""
import io
import os
import sys

import requests
from PIL import Image

DINO_URL = os.environ.get("DINO_URL", "http://127.0.0.1:8000")
BLIP2_URL = os.environ["BLIP2_URL"]
BLIP2_TOKEN = os.environ["BLIP2_TOKEN"]
PADDING = 0.15   # keep a little of the surroundings around each object


def to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def detect(img, prompts):
    r = requests.post(f"{DINO_URL}/detect", files={"image": to_bytes(img)},
                      data={"prompts": prompts}, timeout=60)
    r.raise_for_status()
    return r.json()


def describe(img, question=""):
    r = requests.post(f"{BLIP2_URL}/describe", files={"image": to_bytes(img)},
                      data={"question": question}, headers={"x-token": BLIP2_TOKEN}, timeout=120)
    r.raise_for_status()
    return r.json()


def crop(img, box, padding=PADDING):
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * padding, (y1 - y0) * padding
    return img.crop((max(0, x0 - px), max(0, y0 - py), min(img.width, x1 + px), min(img.height, y1 + py)))


if __name__ == "__main__":
    photo, prompts = sys.argv[1], sys.argv[2]
    img = Image.open(photo).convert("RGB")
    detections = detect(img, prompts)
    print(f"Grounding DINO found {len(detections)} objects")
    for d in detections:
        piece = crop(img, d["box"])
        caption = describe(piece)
        print(f'- {d["label"]:<20} score {d["score"]:.2f}  ->  "{caption["text"]}" ({caption["confidence"]:.2f})')
