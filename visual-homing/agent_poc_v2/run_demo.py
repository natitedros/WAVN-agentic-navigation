"""
Run:  python run_demo.py

Scenario 1: robot_A asks for the target; everything goes to plan.
Scenario 2: same, but the blue door turns out to be blocked when robot_A gets there,
            so the path finder has to re-plan.
"""
from comms import MessageBus
from path_finder import PathFinderAgent
from robot import FakeNavigator, FakePerceiver, RobotAgent
from scenario import SCENES, TARGET


def run(title, lost_landmarks=()):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    bus = MessageBus()
    bus.register(PathFinderAgent(robots=SCENES))
    for name, scene in SCENES.items():
        navigator = FakeNavigator(lost_landmarks if name == "robot_A" else ())
        bus.register(RobotAgent(name, FakePerceiver(name, scene), navigator))

    bus.agents["robot_A"].request_homing(TARGET, task_id="homing-task-001")
    bus.run()
    print(f"\n  {len(bus.log)} messages in total.")


if __name__ == "__main__":
    run("SCENARIO 1: robot_A wants to reach a target only robot_D can see")
    run("SCENARIO 2: the blue door is blocked when robot_A arrives", lost_landmarks={"C-obj-1"})
