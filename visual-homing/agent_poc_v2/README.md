# Agent architecture proof-of-concept

Plain Python, no libraries to install, no models. Run it with:

    python run_demo.py

## The files

| File | What it is |
|---|---|
| `comms.py` | The message format and a message bus that stands in for the network |
| `matching.py` | How two descriptions are compared (simple key-word overlap for now) |
| `robot.py` | A robot: perception agent + navigator. `FakePerceiver` stands in for Grounding DINO + BLIP-2, `FakeNavigator` for ViNT |
| `path_finder.py` | The shared path-finding agent: collects reports, links robots, asks clarifying questions, picks the chain, re-plans |
| `scenario.py` | The four-robot scene (what each robot "sees") |
| `run_demo.py` | Runs two scenarios and prints every message plus the path finder's reasoning |

## What an "agent" is here

An agent is just an object that:
1. receives messages (`handle` -> `on_<message type>`),
2. keeps what it has learned so far (the path finder keeps one `Task` per homing request),
3. decides what to do next and sends messages.

Agents never call each other directly. That is what makes them easy to move onto separate
machines later (swap `MessageBus` for ROS 2 topics).

## Where the real models go

| Fake part | Replace with | Method to keep |
|---|---|---|
| `FakePerceiver.detect()` | Grounding DINO finds objects, BLIP-2 describes each crop | returns a list of `{object_id, description, confidence}` |
| `FakePerceiver.answer()` | BLIP-2 asked "Is this <description>?" for each candidate crop | returns `(picked_id, answer_text, confidence)` |
| `FakeNavigator.drive_to()` | ViNT driving to the hop's goal image until its distance is small | returns `True` (reached) or `False` (lost) |
| `matching.similarity()` | Comparing BLIP-2 text/image embeddings, or a small language model | returns a score from 0 to 1 |

## Things to try

- Make the two doors at robot_C easy to tell apart (give one a different description) and watch the question disappear.
- Raise both stop-sign confidences (robot_A's and robot_C's) to 0.9 and see the path finder switch to the shortcut.
- Add a robot_E that also sees the target, and see which one is chosen.
- Remove the bench from robot_D's scene and check that `no_route_found` is sent.

## Running with the real models

| File | What it is |
|---|---|
| `model_clients.py` | Calls the Grounding DINO server (laptop) and the BLIP-2 server (Colab) |
| `real_perceiver.py` | `RealPerceiver`: same methods as `FakePerceiver`, backed by the real models |
| `scenario_real.py` | **Edit this:** one photo per robot, the object names to look for, and the target |
| `run_real.py` | Runs the agents on your photos (navigation is still `FakeNavigator`) |

1. Start the Grounding DINO server: `uvicorn model_server:app --host 127.0.0.1 --port 8000`
2. Start the BLIP-2 Colab notebook, then `export BLIP2_URL=...` and `export BLIP2_TOKEN=...`
3. Put one photo per robot in `photos/`, and edit `scenario_real.py`
4. `python run_real.py`

The output shows every message, the path finder's reasoning, what each robot reported, and
saves each object's crop in `debug_crops/` so you can check what the models actually saw.
