"""
Step 6: run the agents on every test case in scenarios/ and score them against the answer key.

Needs the same setup as run_real.py: Grounding DINO server running, and the Qwen Colab server
with BLIP2_URL / BLIP2_TOKEN exported.

    python run_benchmark.py              # all cases
    python run_benchmark.py case_003     # just one (good for a first try)

Each case's full agent log goes to bench_logs/<case>.txt. Scores go to benchmark_results.json.

How a link is judged: for each object in a link, take its box in its robot's frame, read the
depth there and turn it into a world position (same maths as build_world_objects.py). The link
is TRUE if both objects land on the same physical object of the answer key.
"""
import contextlib
import copy
import glob
import io
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

import path_finder
from comms import MessageBus
from path_finder import PathFinderAgent
from real_perceiver import RealPerceiver
from robot import FakeNavigator, RobotAgent
from build_world_objects import (box_depth, find_pngs, frame_number, load_depth, load_poses,
                                 main_noun, pixel_to_world)

# Same settings as run_real.py
path_finder.MATCH_MIN = 1.0
path_finder.TARGET_MIN = 0.5

W = json.load(open("world_objects.json"))
TRAJ_DIR = W["traj_dir"]
PROMPTS = W["settings"]["prompts"]
SAME_OBJECT_M = W["settings"]["same_object_m"]
POSES = load_poses(TRAJ_DIR)
DEPTH_FILES = {frame_number(p): p for p in find_pngs(TRAJ_DIR, "depth_lcam_front")}
UIDS_IN = defaultdict(set)
for d in W["detections"]:
    UIDS_IN[d["frame"]].add(d["object_uid"])
WORLD_OBJECTS = [(o["uid"], main_noun(o["label"]), np.array(o["xyz"])) for o in W["objects"]]
LOG_DIR = "bench_logs"


# ---------- perception cache: a frame's detections + Qwen answers are computed once ----------
class CachedPerceiver(RealPerceiver):
    _result = None

    def detect(self, extra_prompts=()):
        if self._result is None:
            self._result = super().detect(extra_prompts)
        return copy.deepcopy(self._result)


_perceivers = {}


def get_perceiver(name, photo, target_label):
    key = (name, photo, target_label)
    if key not in _perceivers:
        _perceivers[key] = CachedPerceiver(name, photo, PROMPTS)
    return _perceivers[key]


# ---------- answer key lookups ----------
_depth_cache = {}


def world_position(frame, box):
    if frame not in _depth_cache:
        _depth_cache[frame] = load_depth(DEPTH_FILES[frame])
    z = box_depth(_depth_cache[frame], box)
    if z is None:
        return None
    x0, y0, x1, y1 = box
    return pixel_to_world((x0 + x1) / 2, (y0 + y1) / 2, z, POSES[frame])


def answer_key_uid(frame, obj):
    """Which physical object of the answer key is this agent object? (None if not in the key.)"""
    p = world_position(frame, obj["bbox"])
    if p is None:
        return None, None
    kind = main_noun(obj["label"])
    best, best_d = None, SAME_OBJECT_M
    for uid, k, xyz in WORLD_OBJECTS:
        d = float(np.linalg.norm(p - xyz))
        if k == kind and d < best_d:
            best, best_d = uid, d
    return best, p


def same_physical_object(fa, oa, fb, ob):
    ua, pa = answer_key_uid(fa, oa)
    ub, pb = answer_key_uid(fb, ob)
    if ua is not None and ua == ub:
        return True, ua
    if pa is not None and pb is not None and main_noun(oa["label"]) == main_noun(ob["label"]) \
            and np.linalg.norm(pa - pb) < SAME_OBJECT_M:
        return True, ua or ub            # same place, just not in the key (e.g. missed by the key's detector)
    return False, None


# ---------- run + score one case ----------
def run_case(case):
    bus = MessageBus()
    bus.register(PathFinderAgent(robots=case["robots"]))
    for name, photo in case["robots"].items():
        bus.register(RobotAgent(name, get_perceiver(name, photo, case["target_label"]), FakeNavigator()))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        bus.agents[case["requester"]].request_homing(case["target_description"], task_id=case["case_id"],
                                                     target_label=case["target_label"])
        bus.run()
    return bus, log.getvalue()


def score_case(case, bus):
    frames = case["frames"]
    pf = bus.agents["path_finder"]
    task = pf.tasks[case["case_id"]]          # if this fails, tell Claude how your PathFinderAgent stores tasks

    # Links
    links = []
    for x, ox, y, oy, conf in task.links:
        ok, uid = same_physical_object(frames[x], ox, frames[y], oy)
        links.append({"robots": [x, y], "objects": [ox["object_id"], oy["object_id"]],
                      "label": ox["label"], "confidence": round(conf, 3), "true": ok, "uid": uid})
    available = set()
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for uid in UIDS_IN[frames[a]] & UIDS_IN[frames[b]]:
                available.add((frozenset((a, b)), uid))
    found = {(frozenset(l["robots"]), l["uid"]) for l in links if l["true"] and l["uid"]}

    # Route
    objects = {}
    for m in bus.log:
        if m.type == "observation_report":
            for o in m.payload["objects"]:
                objects[(m.sender, o["object_id"])] = o
    route = next((m for m in reversed(bus.log) if m.type == "route_instruction"), None)
    route_ok = None
    if route:
        route_ok, previous = True, case["requester"]
        for h in route.payload["hops"]:
            obj = objects[(h["via_robot"], h["landmark_id"])]
            uid, p = answer_key_uid(frames[h["via_robot"]], obj)
            if h.get("is_target"):
                hit = uid == case["target_uid"] or (
                    p is not None and np.linalg.norm(p - np.array(case["target_xyz"])) < SAME_OBJECT_M)
            else:
                hit = uid is not None and uid in UIDS_IN[frames[previous]]
                previous = h["via_robot"]
            route_ok = route_ok and hit

    if case["expected"] == "route":
        outcome = "correct route" if route_ok else ("WRONG route" if route else "missed (no route)")
    else:
        outcome = "correct (no route)" if not route else "FALSE route"
    return {"case_id": case["case_id"], "kind": case["kind"], "outcome": outcome,
            "links_made": len(links), "links_true": sum(l["true"] for l in links),
            "true_links_available": len(available), "true_links_found": len(found & available),
            "links": links}


def main():
    paths = sorted(glob.glob("scenarios/case_*.json"))
    if len(sys.argv) > 1:
        paths = [p for p in paths if any(a in p for a in sys.argv[1:])]
    os.makedirs(LOG_DIR, exist_ok=True)
    results, start = [], time.time()
    for p in paths:
        case = json.load(open(p))
        t0 = time.time()
        bus, log = run_case(case)
        with open(os.path.join(LOG_DIR, case["case_id"] + ".txt"), "w") as f:
            f.write(log)
        r = score_case(case, bus)
        results.append(r)
        print(f"{r['case_id']}  {r['kind']:8}  {r['outcome']:20}  links {r['links_true']}/{r['links_made']} true, "
              f"found {r['true_links_found']}/{r['true_links_available']} available  ({time.time() - t0:.0f}s)")

    made = sum(r["links_made"] for r in results)
    true = sum(r["links_true"] for r in results)
    avail = sum(r["true_links_available"] for r in results)
    found = sum(r["true_links_found"] for r in results)
    pos = [r for r in results if r["kind"] == "positive"]
    neg = [r for r in results if r["kind"] == "negative"]
    summary = {
        "cases": len(results),
        "link_precision": round(true / made, 3) if made else None,
        "link_recall": round(found / avail, 3) if avail else None,
        "route_success": f"{sum(r['outcome'] == 'correct route' for r in pos)}/{len(pos)}",
        "wrong_routes": f"{sum(r['outcome'] == 'WRONG route' for r in pos)}/{len(pos)}",
        "false_routes_on_negatives": f"{sum(r['outcome'] == 'FALSE route' for r in neg)}/{len(neg)}",
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k:26} {v}")
    print(f"  total time                 {time.time() - start:.0f}s")
    json.dump({"summary": summary, "cases": results}, open("benchmark_results.json", "w"), indent=1)
    print("\nSaved benchmark_results.json; per-case agent logs in bench_logs/")


if __name__ == "__main__":
    main()
