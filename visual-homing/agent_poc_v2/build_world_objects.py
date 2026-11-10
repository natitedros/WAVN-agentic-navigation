"""
Step 3-4 of the TartanGround test set.

  Step 3: walk along one trajectory and keep one frame every SPACING_M metres.
          Each kept frame is a "virtual robot" standing at a known position.
  Step 4: run Grounding DINO on each kept frame, and give every detected object
          a position in the world (x, y, z in metres), using depth + camera pose.
          Then group detections that are the same physical object across frames
          (same kind of object, world positions close together) under one object_uid.

Output: world_objects.json  -> the answer key: which frames see which physical object.

Run from your visual-homing folder (so it can import model_clients and real_perceiver),
with the Grounding DINO server running:
    python build_world_objects.py
"""
import glob
import json
import os
import re

import cv2
import numpy as np
from PIL import Image

import model_clients
from real_perceiver import clean_label

# ---------------- settings: edit these ----------------
TRAJ_DIR = "../../Datasets/Antiquity3D/Antiquity3D/Data_diff/P1000"  # folder holding image_lcam_front/
MAX_FRAMES = 700          # only look at the first N frames (None = all)
SPACING_M = 2.0           # keep one frame every this many metres
THRESHOLD = 0.35          # Grounding DINO score cut-off
MAX_DEPTH_M = 40.0        # ignore objects further than this (depth gets unreliable, sky is huge)
SAME_OBJECT_M = 1.5       # two detections of the same kind closer than this = same physical object
PROMPTS = ["column", "statue", "arch", "stone wall", "staircase", "pedestal",
           "fountain", "doorway", "tree", "bench", "vase", "building", "lion statue", "pillar", "carving"]
# For 640x640 images with a 90 degree field of view (check the metadata file if unsure):
FX = FY = 320.0
CX = CY = 320.0
OUT_FILE = "world_objects.json"
# ------------------------------------------------------


def frame_number(path):
    return int(re.match(r"(\d+)", os.path.basename(path)).group(1))


def load_poses(traj_dir):
    """pose_lcam_front.txt: one line per frame: x y z qx qy qz qw (NED world frame)."""
    return np.loadtxt(os.path.join(traj_dir, "pose_lcam_front.txt"))


def load_depth(path):
    """TartanGround depth PNGs store one float32 per pixel in the 4 RGBA bytes."""
    rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return rgba.view("<f4").squeeze(-1)


def quat_to_matrix(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])


def pixel_to_world(u, v, depth, pose):
    """Pixel (u, v) at the given depth -> world (x, y, z), all in metres, NED."""
    # camera image frame: x right, y down, z forward
    x_c = (u - CX) / FX * depth
    y_c = (v - CY) / FY * depth
    z_c = depth
    # TartanGround camera body frame is NED-style: forward, right, down
    p_body = np.array([z_c, x_c, y_c])
    R = quat_to_matrix(*pose[3:7])
    return R @ p_body + pose[:3]


def box_depth(depth, box):
    """Median depth of the middle part of the box (edges often hit the background)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    patch = depth[int(y0 + 0.35 * h):int(y1 - 0.35 * h) + 1, int(x0 + 0.35 * w):int(x1 - 0.35 * w) + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


def main_noun(label):
    return label.lower().split()[-1]


def pick_frames(images, poses):
    """Step 3: one frame every SPACING_M metres along the path."""
    kept, last = [], None
    for path in images:
        n = frame_number(path)
        pos = poses[n, :3]
        if last is None or np.linalg.norm(pos - last) >= SPACING_M:
            kept.append(path)
            last = pos
    return kept


def main():
    images = sorted(glob.glob(os.path.join(TRAJ_DIR, "image_lcam_front", "*.png")), key=frame_number)
    if MAX_FRAMES:
        images = images[:MAX_FRAMES]
    poses = load_poses(TRAJ_DIR)
    depth_dir = os.path.join(TRAJ_DIR, "depth_lcam_front")
    depth_files = {frame_number(p): p for p in glob.glob(os.path.join(depth_dir, "*.png"))}
    if not depth_files:
        raise SystemExit(f"No depth images in {depth_dir}. Download depth_lcam_front for this trajectory first.")

    frames = pick_frames(images, poses)
    print(f"{len(images)} frames on disk -> kept {len(frames)} (one every {SPACING_M} m)")

    detections = []
    for i, path in enumerate(frames):
        n = frame_number(path)
        if n not in depth_files:
            print(f"  frame {n}: no depth image, skipped")
            continue
        img = Image.open(path).convert("RGB")
        depth = load_depth(depth_files[n])
        found = model_clients.detect(img, PROMPTS, threshold=THRESHOLD)
        kept_here = 0
        for d in found:
            label = clean_label(d["label"], PROMPTS)
            if not label:
                continue
            z = box_depth(depth, d["box"])
            if z is None or z > MAX_DEPTH_M:
                continue
            x0, y0, x1, y1 = d["box"]
            world = pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2, z, poses[n])
            detections.append({"frame": n, "image": path, "label": label, "score": round(d["score"], 3),
                               "box": [round(v, 1) for v in d["box"]], "depth_m": round(z, 2),
                               "world_xyz": [round(float(v), 2) for v in world]})
            kept_here += 1
        print(f"  [{i + 1}/{len(frames)}] frame {n}: {kept_here} objects")

    # Step 4b: same kind + close in the world = same physical object
    objects = []          # each: {"uid", "label", "xyz" (running average), "count"}
    for d in sorted(detections, key=lambda d: -d["score"]):
        p = np.array(d["world_xyz"])
        match = None
        for o in objects:
            if main_noun(o["label"]) == main_noun(d["label"]) and np.linalg.norm(p - o["xyz"]) < SAME_OBJECT_M:
                match = o
                break
        if match is None:
            match = {"uid": f"obj-{len(objects) + 1}", "label": d["label"], "xyz": p, "count": 0}
            objects.append(match)
        match["xyz"] = (match["xyz"] * match["count"] + p) / (match["count"] + 1)
        match["count"] += 1
        d["object_uid"] = match["uid"]

    frames_per_obj = {}
    for d in detections:
        frames_per_obj.setdefault(d["object_uid"], set()).add(d["frame"])
    shared = {uid: f for uid, f in frames_per_obj.items() if len(f) >= 2}

    print(f"\n{len(detections)} detections -> {len(objects)} physical objects")
    print(f"{len(shared)} objects are seen from 2 or more frames (possible shared landmarks)")
    by_label = {}
    for o in objects:
        by_label[o["label"]] = by_label.get(o["label"], 0) + 1
    print("objects per kind:", dict(sorted(by_label.items(), key=lambda kv: -kv[1])))

    out = {
        "traj_dir": TRAJ_DIR,
        "settings": {"spacing_m": SPACING_M, "threshold": THRESHOLD, "same_object_m": SAME_OBJECT_M,
                     "prompts": PROMPTS},
        "frames": [{"frame": frame_number(p), "image": p,
                    "position": [round(float(v), 2) for v in poses[frame_number(p), :3]]} for p in frames],
        "objects": [{"uid": o["uid"], "label": o["label"], "xyz": [round(float(v), 2) for v in o["xyz"]],
                     "frames": sorted(frames_per_obj[o["uid"]])} for o in objects],
        "detections": detections,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved {OUT_FILE}. Next: python check_world_objects.py")


if __name__ == "__main__":
    main()
