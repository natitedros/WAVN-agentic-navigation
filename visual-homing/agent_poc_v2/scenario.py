"""
The four-robot scene from the architecture diagram.

A and B both see a mailbox, B and C both see a blue door, C and D both see a bench,
and only D sees the target. Two twists make the agents work for it:
  - C sees TWO blue doors that BLIP-2 would describe the same way, so the path finder
    has to ask which one is "next to a metal fence".
  - A and C both see a stop sign, a shortcut with fewer hops but a weak match.
"""
TARGET = "a red fire extinguisher on a white wall"

SCENES = {
    "robot_A": [
        {"description": "a yellow mailbox", "confidence": 0.90},
        {"description": "a parked white car", "confidence": 0.85},
        {"description": "a stop sign", "confidence": 0.35},           # far away, hard to see
    ],
    "robot_B": [
        {"description": "a yellow mailbox by the road", "confidence": 0.85},
        {"description": "a blue door next to a metal fence", "confidence": 0.88},
    ],
    "robot_C": [
        {"description": "a blue door", "confidence": 0.80, "facts": ["next to a metal fence"]},
        {"description": "a blue door", "confidence": 0.82, "facts": ["next to a window"]},
        {"description": "a red stop sign", "confidence": 0.60},
        {"description": "a wooden bench under a tree", "confidence": 0.90},
    ],
    "robot_D": [
        {"description": "a wooden bench under a tree", "confidence": 0.85},
        {"description": TARGET, "confidence": 0.92},
    ],
}
