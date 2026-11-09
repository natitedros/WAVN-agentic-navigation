"""
A real-photo scenario. EDIT THIS FILE for your own photos.

Each robot is one photo. For a chain to exist, neighbouring robots' photos must share at
least one visible object: for example, frames from one TartanGround route taken a few
meters apart (robot_A = frame 100, robot_B = frame 160, robot_C = frame 220, ...).

TARGET should describe something visible ONLY in the last robot's photo, in words close to
how BLIP-2 would describe it.
"""
PHOTO_FOLDER = "photos"     # relative to agent_poc/, or an absolute path

ROBOT_PHOTOS = {
    "robot_A": "robot_A.png",
    "robot_B": "robot_B.png",
    "robot_C": "robot_C.png",
    "robot_D": "robot_D.png",
}

# Object names Grounding DINO looks for: things that appear in your environment

# 4 coffey field images
PROMPTS = [
    "light pole",
    "stadium floodlight",
    "padded wall",
    "metal fence",
    "hurdle",
    "concrete block",
    "metal grate",
    "wooden board on stand",
    "weight sled",
    "batting cage",
    "netting",
    "pitching machine",
    "hedge",
    "sled",
    "scoreboard",
]

# 2 coffey field images
# PROMPTS = ["light pole", "concrete block", "hurdle", "padded wall", "metal fence", "hedge", "tree", "small goal frame"]

TARGET_LABEL = "circular target"
TARGET = "orange and black circular target on black wall"
# Object names Grounding DINO always looks for. Keep them distinct: no two names that
# share a word ("statue" and "lion statue"), or Grounding DINO merges them.
# PROMPTS = ["a column", "a brick wall", "a statue", "an arch", "a staircase", "a tree", "a rock"]

# # What kind of object the target is (added to every robot's PROMPTS for this request),
# # and a short description of it in the same plain words.
# TARGET_LABEL = "a relief carving"
# TARGET = "a golden relief carving of a woman"