"""
One robot = a perception agent plus a navigator, running on the robot itself.

The two "model" parts are swappable:
  - FakePerceiver  stands in for Grounding DINO + BLIP-2
  - FakeNavigator  stands in for ViNT
Later, write RealPerceiver / RealNavigator with the same methods and nothing else changes.
"""
from comms import Agent
from matching import key_words


class FakePerceiver:
    """Pretends to see a fixed scene. Each object has what the camera models would report,
    plus hidden 'facts' that BLIP-2 could only discover when asked a question."""

    def __init__(self, robot, scene):
        self.robot = robot
        self.scene = scene   # list of dicts: description, confidence, facts

    def detect(self, extra_prompts=()):
        # Real version: Grounding DINO finds objects, BLIP-2 describes each crop.
        # (extra_prompts is only used by the real version.)
        return [{"object_id": f"{self.robot[-1]}-obj-{i}",
                 "description": o["description"],
                 "confidence": o["confidence"]}
                for i, o in enumerate(self.scene, start=1)]

    def answer(self, object_ids, query):
        # Real version: BLIP-2 is shown each candidate crop and asked "Is this <query>?"
        wanted = key_words(query)
        scores = {}
        for i, o in enumerate(self.scene, start=1):
            oid = f"{self.robot[-1]}-obj-{i}"
            if oid in object_ids:
                known = key_words(o["description"]) | key_words(" ".join(o.get("facts", [])))
                scores[oid] = len(wanted & known) / len(wanted)
        best = max(scores, key=scores.get)
        if scores[best] < 0.5:
            return None, "none of them", 0.0
        return best, f"{best} matches", round(0.9 * scores[best], 2)


class FakeNavigator:
    """Pretends to drive to a landmark. Returns False for landmarks listed as 'lost'
    (for example, something now blocks the view), to test re-planning."""

    def __init__(self, lost_landmarks=()):
        self.lost = set(lost_landmarks)

    def drive_to(self, hop):
        # Real version: run ViNT with the hop's goal image until its predicted distance is small.
        return hop["landmark_id"] not in self.lost


class RobotAgent(Agent):
    def __init__(self, name, perceiver, navigator):
        super().__init__(name)
        self.perceiver = perceiver
        self.navigator = navigator

    # --- this robot asks for help -------------------------------------------------
    def request_homing(self, target_description, task_id, target_label=None):
        # target_label: what kind of object the target is ("a relief carving"), if known
        payload = {"target_description": target_description}
        if target_label:
            payload["target_label"] = target_label
        self.send("homing_request", "path_finder", payload, task_id=task_id)

    # --- perception agent: answering the path finder ------------------------------
    def on_observation_request(self, msg):
        # Also look for the target's kind, so the target itself gets detected
        label = msg.payload.get("target_label")
        objects = self.perceiver.detect(extra_prompts=[label] if label else [])
        self.send("observation_report", "path_finder", {"objects": objects},
                  task_id=msg.task_id, reply_to=msg.message_id)

    def on_clarification_question(self, msg):
        p = msg.payload
        picked, text, conf = self.perceiver.answer(p["about_object_ids"], p["query"])
        self.send("clarification_answer", "path_finder",
                  {"selected_object_id": picked, "answer": text, "confidence": conf},
                  task_id=msg.task_id, reply_to=msg.message_id)

    # --- navigation: following a route --------------------------------------------
    def on_route_instruction(self, msg):
        for hop in msg.payload["hops"]:
            ok = self.navigator.drive_to(hop)
            status = "reached" if ok else "lost_landmark"
            self.send("progress_update", "path_finder",
                      {"hop": hop["hop"], "status": status, "landmark_id": hop["landmark_id"],
                       "is_target": hop.get("is_target", False)},
                      task_id=msg.task_id, reply_to=msg.message_id)
            if not ok:
                break   # stop and wait for a new route
