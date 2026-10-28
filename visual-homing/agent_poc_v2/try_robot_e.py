# Robot_E is one hop from robot_A but barely sees the target. The path finder should still prefer robot_D.
import run_demo
# robot_E is only one hop from robot_A, but sees the target poorly (e.g. far away, half hidden)
run_demo.SCENES["robot_E"] = [
    {"description": "a yellow mailbox on a post", "confidence": 0.80},
    {"description": run_demo.TARGET, "confidence": 0.20},
]
run_demo.run("robot_E close by but barely sees the target")
