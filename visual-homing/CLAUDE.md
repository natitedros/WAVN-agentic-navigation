# WAVN agentic visual homing: project notes for Claude

## What this project is
Multi-robot visual homing with AI agents, extending the DYSWIS / Federated Visual Homing work.
A robot that cannot see its target asks for help. Other robots report what they see, and a shared
path-finder agent builds a chain of robots linked by shared landmarks that reaches a robot that
sees the target. When a landmark match is ambiguous, the path finder asks robots clarifying
questions (the "negotiation"). All models are pretrained; nothing is trained per environment.

- Perception agent: runs on each robot. Grounding DINO finds objects; a vision-language model
  (now Qwen2.5-VL-3B; BLIP-2 before) answers questions about each crop (color, material, yes/no).
- Path-finder agent: one shared agent on a central device. It links objects across robots,
  asks clarifying questions, and picks the most trustworthy chain (Dijkstra on -log confidence,
  with the target as a final node, so the whole path is compared, not just the first hop).
- Navigation: FakeNavigator for now (always "reaches"). ViNT comes later, after matching is reliable.
- Plain Python on purpose (no LangGraph/LangChain), so the architecture stays easy to follow.

## How the user likes to work
- Plain, simple, step-by-step explanations; avoid jargon.
- When changing code, show the exact edits ("replace this with this") or make them directly;
  don't hand over a zip of the whole project.
- Save run logs under clear names (e.g. pair1_qwen_oneprompt.txt). They become the results table.
- Change one thing at a time between comparison runs, so results stay comparable.

## Code (agent_poc_v2/)
- comms.py: Message dataclass and MessageBus. Envelope: message_id, task_id, reply_to, type,
  sender, receiver, payload. Message types: homing_request, observation_request/report,
  clarification_question/answer, route_instruction, progress_update, no_route_found.
  (search_request is designed but not built.)
- robot.py: RobotAgent, FakePerceiver, FakeNavigator.
- real_perceiver.py: RealPerceiver. Runs Grounding DINO one prompt at a time, keeping the
  higher-scoring box when two prompts box the same thing; crops each box with padding; asks
  "What color is the {label}?" and "What is the {label} made of?". short_answer() strips filler
  words, the "'s" contraction and the label's own words, but keeps the label word when nothing else
  is left ("concrete"). only_colors() keeps real color words only. answer() asks "Is this {query}?".
- matching.py: key_words, Jaccard similarity, object_similarity(). Field-wise comparison;
  divides by the number of fields asked, so an unanswered field counts as not agreeing.
- path_finder.py:
  - same_kind() compares the main noun (last word of the label), so "metal fence" != "metal grate".
  - uniqueness() returns 1 / (the most same-kind objects on either robot). Link confidence is
    multiplied by it: a match on a common object (light pole, building) proves little.
  - LINK_MIN = 0.25: links below this, after weighting, are dropped. This applies in link_pair's
    final block and in on_clarification_answer (written as if/else, so the "all answers in ->
    plan route" step still runs).
  - The ambiguity loop skips asking when cands[0][2] * uniqueness * obj confidence < LINK_MIN.
  - ANSWER_MIN = 0.4: a robot's "yes" below this counts as "none". Raise to about 0.7 for Qwen.
  - Link confidence = similarity * min(DINO scores) * uniqueness. Target confidence =
    DINO score * the robot's yes-probability. Route confidence = product over hops.
- run_real.py: one scenario from scenario_real.py (PHOTO_FOLDER, ROBOT_PHOTOS, PROMPTS, TARGET,
  TARGET_LABEL). Sets MATCH_MIN = 1.0 (all fields must agree). The first robot listed is the
  requester. Prints the agent log, each robot's objects, and the final route.
- model_clients.py: detect() -> DINO_URL/detect (default http://127.0.0.1:8000);
  describe() -> BLIP2_URL/describe with header x-token = BLIP2_TOKEN.

## Running the model servers
- Grounding DINO (tiny, locally on the user's Ubuntu laptop, RTX 2060 6 GB):
  `uvicorn model_server:app --host 127.0.0.1 --port 8000`
- Qwen2.5-VL-3B on a Colab T4: qwen_server.ipynb (float16). It serves the same /describe endpoint
  as the old BLIP-2 server and prints the two lines to export:
  `export BLIP2_URL=...` (no trailing slash) and `export BLIP2_TOKEN=...`.
  The variable names stayed BLIP2_*, so client code is unchanged. The token is a password: keep it out of git.
- test_servers.py checks both servers.

## Results so far (own photos, sports-complex at night)
- 2-robot pair, shared concrete block: BLIP-2 gave no route ("shade brown" vs "light gray");
  Qwen gave the correct route (overall 0.41). Same detections, so the only change was the model.
- Negative pair (nothing shared): no route, which is correct.
- 4 robots with Qwen: first a false A->D route through two different light poles, plus a
  "metal fence" = "metal grate" link. After the main-noun check and distinctiveness weighting:
  no route. That is correct, because A's real landmarks were never detected.
- Remaining weaknesses: Grounding DINO mislabels objects (a board as "concrete block", hurdles as
  "small goal frame") and misses some landmarks; repeated look-alike objects; exact word overlap
  misses synonyms ("buxus sempervirens" vs "plants").

## TartanGround benchmark (automatic answer key)
Development set: Antiquity3D, Data_diff/P1000. Folder layout under the route:
image_lcam_front/, depth_lcam_front/, seg_lcam_front/, and pose_lcam_front.txt from metadata.zip.
Folders may be nested (image_lcam_front/image_lcam_front/); the scripts handle that.
Images are 640x640 with a 90° field of view: fx = fy = cx = cy = 320.
Poses are NED (x y z qx qy qz qw). Depth PNGs are float32 packed in RGBA: `.view("<f4")`.

1. build_world_objects.py: keeps one frame every 2 m (MAX_FRAMES = 700). Runs DINO in one pass
   and takes the median depth of the box centre, giving a world xyz. Same main noun within 2 m
   = same object_uid. Writes world_objects.json.
   Run results: 45 frames, 132 objects, 45 seen from 2+ frames.
2. check_world_objects.py: crop strips per object in check_objects/. Eyeball check: about 2 odd
   crops in ~50.
3. make_scenarios.py: 20 positive cases (requester can't see the target; a true chain of
   1-3 hops; 1-2 distractors; robots shuffled) and 10 negative cases (no true chain). Targets are
   rare kinds (at most 5 instances). Robots are named robot_f<frame>. Writes scenarios/case_XXX.json.
4. run_benchmark.py [case_id]: runs the real agents (Qwen, MATCH_MIN = 1.0) and caches each frame's
   perception. Judges each link by the world positions of both boxes. Reports link precision,
   link recall, route success, wrong routes, and false routes on negatives.
   Logs go to bench_logs/; results to benchmark_results.json.
   Not yet run on real data. Possible fixes:
   - The scorer reads `pf.tasks[case_id]`; check the attribute name in PathFinderAgent.
   - RealPerceiver may need `save_crops_to=None`.

## Next steps (in order)
1. Run make_scenarios.py, then run_benchmark.py on case_001, then on all cases. This is the baseline.
2. Choose the main TartanGround environment: outdoor, man-made objects Grounding DINO can name,
   a mix of unique and repeated objects, and several trajectories (different viewpoints). Rerun the pipeline there.
3. Add improvements one at a time and re-score after each:
   - Spatial context: match small groups of objects using box geometry (left of, in front of).
   - Two-image verification: send the other robot's crop and ask Qwen "Are these the same object?".
     This is the planned centre of the "negotiation" contribution.
   - Image-embedding similarity (DINOv2 or CLIP) on the crops.
   - Whole-view overlap check (place recognition) before linking.
4. Later: a RealNavigator with ViNT (weights vint.pth; 85x64 input, context 5; earlier tests:
   straight driving fine, turns untested), the search_request message, and a stronger "is this the target" check.
