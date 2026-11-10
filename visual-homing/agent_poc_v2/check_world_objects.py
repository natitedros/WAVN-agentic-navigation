"""
Eyeball check for world_objects.json.

For the objects seen in the most frames, saves one picture per object with its crop
from each frame side by side. If every crop in a strip shows the SAME physical object,
the world positions (and so the answer key) can be trusted.

    python check_world_objects.py
Pictures go to check_objects/.
"""
import json
import os

from PIL import Image

N_OBJECTS = 8        # how many objects to check
CROPS_PER_OBJECT = 6
OUT_DIR = "check_objects"

data = json.load(open("world_objects.json"))
os.makedirs(OUT_DIR, exist_ok=True)

objects = sorted(data["objects"], key=lambda o: -len(o["frames"]))[:N_OBJECTS]
for o in objects:
    dets = [d for d in data["detections"] if d["object_uid"] == o["uid"]]
    dets = sorted(dets, key=lambda d: d["frame"])
    step = max(1, len(dets) // CROPS_PER_OBJECT)
    dets = dets[::step][:CROPS_PER_OBJECT]
    crops = []
    for d in dets:
        img = Image.open(d["image"]).convert("RGB")
        x0, y0, x1, y1 = d["box"]
        crops.append(img.crop((int(x0), int(y0), int(x1), int(y1))).resize((200, 200)))
    strip = Image.new("RGB", (200 * len(crops), 200), "white")
    for i, c in enumerate(crops):
        strip.paste(c, (200 * i, 0))
    name = f"{o['uid']}_{o['label'].replace(' ', '_')}.png"
    strip.save(os.path.join(OUT_DIR, name))
    print(f"{o['uid']:8} {o['label']:12} seen in {len(o['frames']):3} frames  at {o['xyz']}  -> {name}")

print(f"\nOpen the pictures in {OUT_DIR}/. Each strip should show one and the same object.")
