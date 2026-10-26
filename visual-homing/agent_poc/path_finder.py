"""
The shared path-finding agent (runs on the central device).

What it does for one homing request:
  1. Ask every robot what it sees                (observation_request)
  2. Find which robots see the target
  3. Link robots that see the same landmark      (the landmark graph)
     - if a match is ambiguous, ask a clarifying question first
  4. Pick the most trustworthy chain of links    (route_instruction)
  5. Follow progress; if a landmark is lost, remove that link and re-plan
"""
import heapq
import math
from collections import deque

from comms import Agent
from matching import similarity

MATCH_MIN = 0.3          # below this, two descriptions are not the same thing
TARGET_MIN = 0.5         # how close a description must be to the target description
AMBIGUITY_MARGIN = 0.1   # two candidates closer than this = ask a question


class Task:
    def __init__(self, task_id, requester, target):
        self.task_id = task_id
        self.requester = requester
        self.target = target
        self.waiting_for = set()   # robots we still expect a report from
        self.reports = {}          # robot -> list of objects it sees
        self.links = []            # (robot_x, obj_x, robot_y, obj_y, confidence)
        self.target_hits = {}      # robot -> (object, confidence)
        self.questions = {}        # message_id -> what the question was about
        self.position = requester  # whose area the requesting robot is in now


class PathFinderAgent(Agent):
    def __init__(self, robots):
        super().__init__("path_finder")
        self.robots = list(robots)     # robots known to be online (in a real system: from heartbeats)
        self.tasks = {}

    # 1. A robot asks for help --------------------------------------------------------
    def on_homing_request(self, msg):
        task = Task(msg.task_id, msg.sender, msg.payload["target_description"])
        self.tasks[msg.task_id] = task
        task.waiting_for = set(self.robots)
        self.think(f"{msg.sender} wants to reach \"{task.target}\". Asking all {len(self.robots)} robots what they see.")
        self.send("observation_request", "all", {"target_description": task.target}, task_id=task.task_id)

    def on_observation_report(self, msg):
        task = self.tasks[msg.task_id]
        task.reports[msg.sender] = msg.payload["objects"]
        task.waiting_for.discard(msg.sender)
        if not task.waiting_for:
            self.build_graph(task)

    # 2 + 3. Find the target and link robots through shared landmarks -----------------
    def build_graph(self, task):
        self.think("All reports in. Looking for the target and for shared landmarks.")
        for robot, objects in task.reports.items():
            for obj in objects:
                score = similarity(obj["description"], task.target)
                if score >= TARGET_MIN:
                    task.target_hits[robot] = (obj, score * obj["confidence"])
                    self.think(f"{robot} sees the target: \"{obj['description']}\" (match {score:.2f})")

        names = sorted(task.reports)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                self.link_pair(task, x, y)

        if not task.questions:
            self.plan_route(task)

    def link_pair(self, task, x, y):
        pairs = [(ox, oy, similarity(ox["description"], oy["description"]))
                 for ox in task.reports[x] for oy in task.reports[y]]
        pairs = [p for p in pairs if p[2] >= MATCH_MIN]
        unsure = set()

        # One object in x matches two objects in y about equally well -> ask y (and the reverse)
        for asker, other, side in ((x, y, 0), (y, x, 1)):
            for obj in task.reports[asker]:
                cands = sorted((p for p in pairs if p[side] is obj), key=lambda p: -p[2])
                if len(cands) >= 2 and cands[0][2] - cands[1][2] < AMBIGUITY_MARGIN:
                    options = [p[1 - side] for p in cands]
                    self.think(f"\"{obj['description']}\" ({asker}) matches {len(options)} objects in {other} "
                               f"equally well ({cands[0][2]:.2f} vs {cands[1][2]:.2f}). Asking {other} to clarify.")
                    q = self.send("clarification_question", other,
                                  {"about_object_ids": [o["object_id"] for o in options],
                                   "query": obj["description"],
                                   "question": f"Which of these is \"{obj['description']}\"?"},
                                  task_id=task.task_id)
                    task.questions[q.message_id] = (asker, obj, other, options)
                    unsure.update(id(p) for p in cands)

        # Everything else: accept a pair if each is the other's best match
        for ox, oy, score in pairs:
            if any(id(p) in unsure for p in pairs if p[0] is ox and p[1] is oy):
                continue
            best_for_x = max((p for p in pairs if p[0] is ox), key=lambda p: p[2])
            best_for_y = max((p for p in pairs if p[1] is oy), key=lambda p: p[2])
            if best_for_x[1] is oy and best_for_y[0] is ox:
                conf = score * min(ox["confidence"], oy["confidence"])
                task.links.append((x, ox, y, oy, conf))
                self.think(f"Link {x} <-> {y}: \"{ox['description']}\" = \"{oy['description']}\" (confidence {conf:.2f})")

    def on_clarification_answer(self, msg):
        task = self.tasks[msg.task_id]
        asker, obj, other, options = task.questions.pop(msg.reply_to)
        picked = msg.payload["selected_object_id"]
        if picked:
            chosen = next(o for o in options if o["object_id"] == picked)
            conf = msg.payload["confidence"] * min(obj["confidence"], chosen["confidence"])
            task.links.append((asker, obj, other, chosen, conf))
            self.think(f"{other} says {picked}. Link {asker} <-> {other}: \"{obj['description']}\" (confidence {conf:.2f})")
        else:
            self.think(f"{other} could not confirm any of them. No link.")
        if not task.questions:
            self.plan_route(task)

    # 4. Choose the chain -------------------------------------------------------------
    def neighbours(self, task):
        graph = {r: [] for r in task.reports}
        for x, ox, y, oy, conf in task.links:
            graph[x].append((y, conf, oy))   # landmark as the next robot sees it
            graph[y].append((x, conf, ox))
        return graph

    def best_chain(self, task, start):
        """Most trustworthy chain: maximise the product of all confidences on the path, including
        how well the last robot sees the target (Dijkstra on -log, with the target as a final node)."""
        graph = self.neighbours(task)
        queue, seen = [(0.0, start, [])], set()
        while queue:
            cost, robot, hops = heapq.heappop(queue)
            if robot == "TARGET":
                return hops, cost          # the whole path, including how well the last robot sees the target
            if robot in seen:
                continue
            seen.add(robot)
            if robot in task.target_hits:
                obj, conf = task.target_hits[robot]
                if conf > 0:
                    heapq.heappush(queue, (cost - math.log(conf), "TARGET", hops + [(robot, obj, conf)]))
            for nxt, conf, landmark in graph[robot]:
                if nxt not in seen and conf > 0:
                    heapq.heappush(queue, (cost - math.log(conf), nxt, hops + [(nxt, landmark, conf)]))
        return None, None

    def fewest_hops(self, task, start):
        """What plain breadth-first search (as in DYSWIS) would pick, for comparison."""
        graph, queue, seen = self.neighbours(task), deque([[start]]), {start}
        while queue:
            path = queue.popleft()
            if path[-1] in task.target_hits:
                return path
            for nxt, _, _ in graph[path[-1]]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(path + [nxt])
        return None

    def plan_route(self, task):
        chain, cost = self.best_chain(task, task.position)
        if chain is None:
            self.think("No chain of shared landmarks reaches the target.")
            self.send("no_route_found", task.requester, {}, task_id=task.task_id)
            return
        robots = [task.position] + [r for r, _, _ in chain[:-1]]
        self.think(f"Most trustworthy chain: {' -> '.join(robots)} -> target "
                                      f"(overall confidence {math.exp(-cost):.2f})")
        bfs = self.fewest_hops(task, task.position)
        if bfs and bfs != robots:
            self.think(f"(Fewest-hops search would have picked {' -> '.join(bfs)} instead.)")
        hops = []
        for n, (robot, landmark, conf) in enumerate(chain, start=1):
            hops.append({"hop": n, "via_robot": robot, "landmark_id": landmark["object_id"],
                         "landmark_description": landmark["description"],
                         "goal_image_ref": f"img/{robot}/{landmark['object_id']}.jpg",
                         "is_target": n == len(chain)})
        task.route = hops
        self.send("route_instruction", task.requester, {"hops": hops}, task_id=task.task_id)

    # 5. Follow progress and re-plan if needed ---------------------------------------
    def on_progress_update(self, msg):
        task = self.tasks[msg.task_id]
        p = msg.payload
        hop = task.route[p["hop"] - 1]
        if p["status"] == "reached":
            task.position = hop["via_robot"]
            if p["is_target"]:
                self.think(f"{task.requester} reached the target. Task done.")
        elif p["status"] == "lost_landmark":
            lost = p["landmark_id"]
            self.think(f"{task.requester} lost sight of \"{hop['landmark_description']}\". "
                       f"Removing that link and re-planning from {task.position}'s area.")
            task.links = [l for l in task.links if lost not in (l[1]["object_id"], l[3]["object_id"])]
            task.target_hits = {r: h for r, h in task.target_hits.items() if h[0]["object_id"] != lost}
            self.plan_route(task)
