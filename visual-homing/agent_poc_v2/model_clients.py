"""
Small helpers that call the two model servers.

  - Grounding DINO: your laptop,   DINO_URL  (default http://127.0.0.1:8000)
  - BLIP-2:         Colab tunnel,  BLIP2_URL + BLIP2_TOKEN (printed by the Colab notebook)
"""
import io
import os

import requests

DINO_URL = os.environ.get("DINO_URL", "http://127.0.0.1:8000").rstrip("/")
BLIP2_URL = os.environ.get("BLIP2_URL", "").rstrip("/")
BLIP2_TOKEN = os.environ.get("BLIP2_TOKEN", "")


def _jpeg(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def detect(img, prompts, threshold=0.35):
    """Grounding DINO: returns [{label, score, box:[x0,y0,x1,y1]}, ...]"""
    r = requests.post(f"{DINO_URL}/detect", files={"image": _jpeg(img)},
                      data={"prompts": ", ".join(prompts), "threshold": threshold}, timeout=60)
    r.raise_for_status()
    return r.json()


def describe(img, question=""):
    """BLIP-2: a caption (no question) or an answer to the question. Returns {text, confidence}."""
    if not BLIP2_URL:
        raise RuntimeError("Set BLIP2_URL and BLIP2_TOKEN (printed by the Colab notebook) first.")
    r = requests.post(f"{BLIP2_URL}/describe", files={"image": _jpeg(img)},
                      data={"question": question}, headers={"x-token": BLIP2_TOKEN}, timeout=120)
    r.raise_for_status()
    return r.json()
