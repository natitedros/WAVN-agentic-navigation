"""
Step 5: turn world_objects.json into test cases for the agents.

Every frame kept by build_world_objects.py can be a "robot". Two frames truly share a
landmark if they both saw the same physical object (same object_uid). From that we build:

  positive cases: a requesting robot that can NOT see the target, plus a chain of robots
                  that truly connects it to a robot that can see the target, plus 1-2
                  unrelated robots as distractors. Right answer: a route.
  negative cases: a requesting robot, a robot that sees the target and 2-3 others, chosen
                  so that NO chain of true shared landmarks connects them. Right answer: no route.

    python make_scenarios.py
Writes scenarios/case_001.json, case_002.json, ...
"""
import json
import os
import random
from collections import Counter, defaultdict, deque

N_POSITIVE = 20
N_NEGATIVE = 10
RARE_MAX = 5          # a target's kind must occur at most this many times in the scene
MIN_HOPS, MAX_HOPS = 1, 3   # robots between requester and target robot: 0 to 2 (hops = links)
SEED = 7
OUT_DIR = "scenarios"


def main_noun(label):
    return label.lower().split()[-1]


def robot_name(frame):
    return f"robot_f{frame:04d}"


def shortest_chain(start, goals, allowed, neighbors):
    """BFS from start to any goal frame, using only frames in `allowed`. Returns the frame list or None."""
    prev, queue = {start: None}, deque([start])
    while queue:
        f = queue.popleft()
        if f in goals:
            chain = []
            while f is not None:
                chain.append(f)
                f = prev[f]
            return chain[::-1]
        for n in neighbors[f]:
            if n in allowed and n not in prev:
                prev[n] = f
                queue.append(n)
    return None


def main():
    random.seed(SEED)
    W = json.load(open("world_objects.json"))
    images = {f["frame"]: f["image"] for f in W["frames"]}
    labels = {o["uid"]: o["label"] for o in W["objects"]}
    uids_in = defaultdict(set)
    for d in W["detections"]:
        uids_in[d["frame"]].add(d["object_uid"])
    all_frames = set(images)
    kinds_in = {f: {main_noun(labels[u]) for u in uids_in[f]} for f in all_frames}
    kind_count = Counter(main_noun(o["label"]) for o in W["objects"])

    neighbors = defaultdict(set)          # frames that truly share at least one object
    for o in W["objects"]:
        for a in o["frames"]:
            for b in o["frames"]:
                if a != b:
                    neighbors[a].add(b)

    targets = [o for o in W["objects"] if kind_count[main_noun(o["label"])] <= RARE_MAX]
    print(f"{len(targets)} possible targets (kinds seen at most {RARE_MAX} times): "
          f"{sorted({t['label'] for t in targets})}")

    def make_case(kind, target, robots_frames, chain):
        requester = robots_frames[0]
        return {
            "kind": kind,
            "expected": "route" if kind == "positive" else "no_route",
            "robots": {robot_name(f): images[f] for f in robots_frames},   # requester first
            "frames": {robot_name(f): f for f in robots_frames},
            "requester": robot_name(requester),
            "target_uid": target["uid"],
            "target_label": target["label"],
            "target_description": target["label"],
            "target_xyz": target["xyz"],
            "true_chain": [robot_name(f) for f in chain] if chain else None,
        }

    cases, seen = [], set()
    tries = 0
    while len([c for c in cases if c["kind"] == "positive"]) < N_POSITIVE and tries < 5000:
        tries += 1
        t = random.choice(targets)
        goal = set(t["frames"])
        kind = main_noun(t["label"])
        requesters = [f for f in all_frames if f not in goal and kind not in kinds_in[f] and uids_in[f]]
        if not requesters:
            continue
        r = random.choice(requesters)
        chain = shortest_chain(r, goal, all_frames, neighbors)
        if not chain or not (MIN_HOPS <= len(chain) - 1 <= MAX_HOPS):
            continue
        others = [f for f in all_frames if f not in chain and kind not in kinds_in[f]]
        distractors = random.sample(others, min(len(others), random.choice([1, 2])))
        key = ("p", t["uid"], r)
        if key in seen:
            continue
        seen.add(key)
        middle = chain[1:] + distractors
        random.shuffle(middle)                      # don't hand the agent the chain in order
        cases.append(make_case("positive", t, [r] + middle, chain))

    tries = 0
    while len([c for c in cases if c["kind"] == "negative"]) < N_NEGATIVE and tries < 5000:
        tries += 1
        t = random.choice(targets)
        kind = main_noun(t["label"])
        tf = random.choice(t["frames"])
        pool = [f for f in all_frames if f not in t["frames"] and kind not in kinds_in[f] and uids_in[f]]
        if len(pool) < 3:
            continue
        r, *others = random.sample(pool, random.choice([3, 4]))
        robots_frames = [r] + others + [tf]
        if shortest_chain(r, {tf}, set(robots_frames), neighbors):
            continue                                # a true chain exists -> not a negative case
        key = ("n", t["uid"], r)
        if key in seen:
            continue
        seen.add(key)
        middle = others + [tf]
        random.shuffle(middle)
        cases.append(make_case("negative", t, [r] + middle, None))

    os.makedirs(OUT_DIR, exist_ok=True)
    for i, c in enumerate(cases, 1):
        c["case_id"] = f"case_{i:03d}"
        with open(os.path.join(OUT_DIR, c["case_id"] + ".json"), "w") as f:
            json.dump(c, f, indent=1)
        hops = len(c["true_chain"]) - 1 if c["true_chain"] else "-"
        print(f"{c['case_id']}  {c['kind']:8}  target={c['target_label']:12}  robots={len(c['robots'])}  true hops={hops}")
    print(f"\nWrote {len(cases)} cases to {OUT_DIR}/ "
          f"({sum(c['kind'] == 'positive' for c in cases)} positive, {sum(c['kind'] == 'negative' for c in cases)} negative)")


if __name__ == "__main__":
    main()
