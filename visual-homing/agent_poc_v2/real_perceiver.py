"""
RealPerceiver: the same methods as FakePerceiver, but backed by the real models.

  detect()  -> Grounding DINO finds objects in the robot's photo. For each object, BLIP-2
               answers three short questions about its crop (color, material, what is next
               to it). The robot shares these answers, not a free caption.
  answer()  -> BLIP-2 is asked "Is this <description>?" about each candidate crop;
               the most confident "yes" wins

Because it has the same methods, RobotAgent and the path finder need no changes for it.
"""
import os
import re

from PIL import Image

import model_clients

# The three things each robot reports about every object it sees
QUESTIONS = {
    "color": "What color is the {label}?",
    "material": "What is the {label} made of?",
    "nearby": "What is next to the {label}?",
}
FILLER = {"it", "its", "is", "are", "the", "a", "an", "this", "that", "made", "of", "next", "to",
          "there", "they", "and", "in", "on", "with", "color", "colour", "unknown", "nothing"}


def clean_label(raw, prompts):
    """Grounding DINO sometimes merges prompts ("column drum pedestal"). Pick the prompt that fits best."""
    raw_words = set(re.findall(r"[a-z]+", raw.lower()))
    best, best_len = raw, 0
    for p in prompts:
        words = [w for w in re.findall(r"[a-z]+", p.lower()) if w not in {"a", "an", "the"}]
        if words and set(words) <= raw_words and len(words) > best_len:
            best, best_len = " ".join(words), len(words)
    return best


def short_answer(text, label, max_words=4):
    """Keep only the informative words of a BLIP-2 answer ("The column is white." -> "white")."""
    label_words = set(re.findall(r"[a-z]+", label.lower()))
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if w not in FILLER and w not in label_words]
    return " ".join(words[:max_words]) or None


class RealPerceiver:
    def __init__(self, robot, image_path, prompts, padding=0.15, min_size=32,
                 threshold=0.35, save_crops_to=None):
        self.robot = robot
        self.image_path = image_path
        self.prompts = list(prompts)    # object names Grounding DINO always looks for
        self.padding = padding          # extra space kept around each object (0.15 = 15%)
        self.min_size = min_size        # skip objects smaller than this many pixels
        self.threshold = threshold      # Grounding DINO score cut-off
        self.save_crops_to = save_crops_to
        self.crops = {}                 # object_id -> cropped image, reused by answer()

    def _crop(self, img, box):
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * self.padding, (y1 - y0) * self.padding
        return img.crop((max(0, x0 - px), max(0, y0 - py),
                         min(img.width, x1 + px), min(img.height, y1 + py)))

    def detect(self, extra_prompts=()):
        # The target's kind (from the homing request) is added to the usual list
        prompts = self.prompts + [p for p in extra_prompts if p and p not in self.prompts]
        img = Image.open(self.image_path).convert("RGB")
        detections = model_clients.detect(img, prompts, threshold=self.threshold)

        objects = []
        for d in sorted(detections, key=lambda d: -d["score"]):
            x0, y0, x1, y1 = d["box"]
            if (x1 - x0) < self.min_size or (y1 - y0) < self.min_size:
                continue
            object_id = f"{self.robot[-1]}-obj-{len(objects) + 1}"
            label = clean_label(d["label"], prompts)
            crop = self._crop(img, d["box"])
            self.crops[object_id] = crop
            if self.save_crops_to:
                folder = os.path.join(self.save_crops_to, self.robot)
                os.makedirs(folder, exist_ok=True)
                crop.save(os.path.join(folder, f"{object_id}.jpg"))

            attributes = {}
            for name, question in QUESTIONS.items():
                reply = model_clients.describe(crop, question=question.format(label=label))
                attributes[name] = short_answer(reply["text"], label)

            description = f"{attributes['color'] or ''} {label}".strip()
            if attributes["material"]:
                description += f" made of {attributes['material']}"
            if attributes["nearby"]:
                description += f" next to {attributes['nearby']}"

            objects.append({
                "object_id": object_id,
                "label": label,                    # what Grounding DINO called it
                "attributes": attributes,          # BLIP-2's short answers
                "description": description,        # the answers joined into one line
                "confidence": d["score"],          # how sure Grounding DINO is the object is there
                "bbox": d["box"],
            })
        return objects

    def answer(self, object_ids, query):
        best_id, best_conf = None, 0.0
        for object_id in object_ids:
            reply = model_clients.describe(self.crops[object_id], question=f"Is this {query}?")
            if reply["text"].strip().lower().startswith("yes") and reply["confidence"] > best_conf:
                best_id, best_conf = object_id, reply["confidence"]
        if best_id is None:
            return None, "none of them", 0.0
        return best_id, f"{best_id} matches", round(best_conf, 2)