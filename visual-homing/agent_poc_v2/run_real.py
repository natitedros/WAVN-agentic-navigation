"""
Run the agents with the real models.

Before running:
  1. Start the Grounding DINO server on this laptop:
       uvicorn model_server:app --host 127.0.0.1 --port 8000
  2. Start the BLIP-2 notebook on Colab and copy the address and token it prints:
       export BLIP2_URL=https://....trycloudflare.com
       export BLIP2_TOKEN=....
  3. Put one photo per robot in agent_poc/photos/ and edit scenario_real.py

Then:  python run_real.py
"""
import os

import path_finder
from comms import MessageBus
from path_finder import PathFinderAgent
from real_perceiver import RealPerceiver
from robot import FakeNavigator, RobotAgent
from scenario_real import PHOTO_FOLDER, PROMPTS, ROBOT_PHOTOS, TARGET, TARGET_LABEL

# How close two descriptions must be. Start with the defaults, look at the printed
# descriptions, then tune. Lowering them finds more links but also more false ones.
path_finder.MATCH_MIN = 0.5    # share of fields (color, material, nearby) that must agree
path_finder.TARGET_MIN = 0.5   # only used when no TARGET_LABEL is given

SAVE_CROPS_TO = "debug_crops"   # crops are saved here so you can check what each robot saw


def main():
    bus = MessageBus()
    bus.register(PathFinderAgent(robots=ROBOT_PHOTOS))
    for name, photo in ROBOT_PHOTOS.items():
        perceiver = RealPerceiver(name, os.path.join(PHOTO_FOLDER, photo), PROMPTS,
                                  save_crops_to=SAVE_CROPS_TO)
        bus.register(RobotAgent(name, perceiver, FakeNavigator()))   # ViNT comes in the next step

    first = next(iter(ROBOT_PHOTOS))
    print(f"{first} asks for: \"{TARGET}\"\n")
    bus.agents[first].request_homing(TARGET, task_id="homing-task-001", target_label=TARGET_LABEL)
    bus.run()

    print("\nWhat each robot reported:")
    for msg in bus.log:
        if msg.type == "observation_report":
            print(f"  {msg.sender}:")
            for o in msg.payload["objects"]:
                print(f"    {o['object_id']}  [{o['label']} {o['confidence']:.2f}]  {o['attributes']}")
    print(f"\nCrops saved in {SAVE_CROPS_TO}/")


if __name__ == "__main__":
    main()
