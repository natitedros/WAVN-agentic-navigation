# Agent architecture proof-of-concept

Plain Python, no libraries to install, no models. Run it with:

    python run_demo.py

## The files

| File             | What it is                                                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `comms.py`       | The message format and a message bus that stands in for the network                                                    |
| `matching.py`    | How two descriptions are compared (simple key-word overlap for now)                                                    |
| `robot.py`       | A robot: perception agent + navigator. `FakePerceiver` stands in for Grounding DINO + BLIP-2, `FakeNavigator` for ViNT |
| `path_finder.py` | The shared path-finding agent: collects reports, links robots, asks clarifying questions, picks the chain, re-plans    |
| `scenario.py`    | The four-robot scene (what each robot "sees")                                                                          |
| `run_demo.py`    | Runs two scenarios and prints every message plus the path finder's reasoning                                           |

## What an "agent" is here

An agent is just an object that:

1. receives messages (`handle` -> `on_<message type>`),
2. keeps what it has learned so far (the path finder keeps one `Task` per homing request),
3. decides what to do next and sends messages.

Agents never call each other directly. That is what makes them easy to move onto separate
machines later (swap `MessageBus` for ROS 2 topics).

## Where the real models go

| Fake part                  | Replace with                                                      | Method to keep                                           |
| -------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| `FakePerceiver.detect()`   | Grounding DINO finds objects, BLIP-2 describes each crop          | returns a list of `{object_id, description, confidence}` |
| `FakePerceiver.answer()`   | BLIP-2 asked "Is this <description>?" for each candidate crop     | returns `(picked_id, answer_text, confidence)`           |
| `FakeNavigator.drive_to()` | ViNT driving to the hop's goal image until its distance is small  | returns `True` (reached) or `False` (lost)               |
| `matching.similarity()`    | Comparing BLIP-2 text/image embeddings, or a small language model | returns a score from 0 to 1                              |

## Things to try

- Make the two doors at robot_C easy to tell apart (give one a different description) and watch the question disappear.
  - Tried and it works. However the clarification question triggers if the score of the objects is close when trying to link robots. (the confidence score influences the clarification as much as similar words existing in the object description. More investigation in how the confidence score is accumulated and linking decision is made is needed.)
- Raise both stop-sign confidences (robot_A's and robot_C's) to 0.9 and see the path finder switch to the shortcut.
- Add a robot_E that also sees the target, and see which one is chosen.
  - Tried but even if the confidence score of seeing the target is manipulated, the path-finder chooses the robot that is linked first, which is a problem. This is a greedy approach where the path is decided with the maximum similarity score for every hop and I want it to see the whole path before deciding so as to avoid any surprises along the way.
  - Fix: The path finder must score the complete route, including how well the final robot sees the target, before choosing. An earlier version stopped at the first robot that saw the target and ignored that confidence. Fixed by treating the target as the route’s final node.
- Remove the bench from robot_D's scene and check that `no_route_found` is sent.
