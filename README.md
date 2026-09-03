# Pine Autopilot — Local Neural Joe Learning Loop

[`pine-autopilot.user.js`](./pine-autopilot.user.js) is a local userscript for Pine & Co Cocktail Defense. After you arm it, it runs the whole gameplay loop by itself:

- starts Cocktail Defense as **Joe**;
- skips the intro;
- moves, dashes when health is low, and triggers the ultimate;
- picks upgrade cards using persistent contextual learning;
- confirms crafts, acknowledges unlocks, and clicks **Cheers**;
- clicks **Retry** after every run and starts Joe again.
- starts the finale chase with **START RUNNING**, then selects **AFTER-HOURS · HELL** automatically.

It uses a small dueling Double Deep Q-Network (DQN): visible HUD data, cooldown state, phase state, four recent 3×3 canvas-perception maps, colour-cluster positions/densities for hostile, hazard, and loot-like pixels, and local motion/danger summaries are fed into neural action-value models for movement, dash/ultimate use, and cards. The movement network produces all 19 action values in one pass. Training uses decision-level 4-step returns, explicit terminal penalties, a 24,000-item sum-tree prioritized replay buffer with importance weighting, frozen target networks, and a retained buffer of elite survival trajectories.

Every same-origin bot window shares its experience samples through `BroadcastChannel`. One automatically elected tab is the central trainer; the rest are experience workers. Each worker has a stable orbit, sweep, ability, or cautious profile, plus a different exploration rate and decision rhythm, so its traces are not duplicates. New or lagging tabs receive a periodic model snapshot. The curriculum has separate Early (0–5m), Mid (5–15m), Late (15m+), and Hell neural heads.

v8 is designed around the actual bottleneck: preventing a noisy stream of early deaths from overwriting the rare policies that reached late game. It keeps a **frozen champion** and trains a separate challenger. Every sixth run is a no-exploration evaluation; v8 alternates the challenger and champion. It promotes a challenger only after five clean evaluations for each policy show a challenger median at least **3% higher**. If five clean challenger evaluations fall below **85%** of the champion median, it automatically restores the champion and clears ordinary replay. A manual restore makes the same recoverable change.

Training remains quality-filtered: a movement/card transition is rejected whenever sampling spans more than **1.75 game seconds**. A full evaluation remains valid when dropped transitions are at most **10% of estimated decisions**. The new replay sampler targets the whole curriculum (Early 30%, Mid 30%, Late 23%, Hell 17%) whenever those examples exist and requests elite survival examples 45% of the time. Movement rewards emphasize survival, the 5-minute/15-minute milestones, and entering Hell; incidental combat contributes only a small signal.

The v8 model is stored only in this browser under `pine-autopilot:joe:neural:v8`. Its first launch imports the compatible v7 challenger and champion, then writes to its own store. The standard replay buffer is RAM-only and rebuilds from live shared traffic after a page reload; elite trajectories and the champion persist locally. **Download training data** exports recent and elite transitions as local JSON. **Download MPS checkpoint** exports the current challenger weights. Neither action transmits data anywhere.

To inspect an export locally, run:

```sh
node offline-analysis.mjs pine-autopilot-training-<timestamp>.json
```

The report shows candidate survival trend, live tournament progress, phase balance, and the best sampled action/phase combinations.

## Offline Apple-silicon challenger training

v8 can use your M4 Pro GPU locally, without pretending that a synthetic game simulator is real. It trains only on recorded Pine transitions, then lets the live game run a blind tournament before accepting the result.

1. In the **central learner** tab, open Neural learner and click **Download training data**, then **Download MPS checkpoint**.
2. In this folder, run:

```sh
python offline-train-mps.py \
  ~/Downloads/pine-autopilot-checkpoint-<timestamp>.json \
  ~/Downloads/pine-autopilot-training-<timestamp>.json \
  --output ~/Downloads/pine-autopilot-challenger.json
```

3. If the script says `Device: mps`, it is using Apple Silicon acceleration. If it says `cpu`, install a current PyTorch build with MPS support and run it again.
4. Back in the **central learner** tab, click **Import MPS challenger** and select `pine-autopilot-challenger.json`.

Importing clears only ordinary replay, advances the tournament generation, and never overwrites the frozen champion. Do not judge an offline pass by its training loss: wait for five clean candidate and five clean champion evaluations. The browser promotes only a measured improvement.

The **Restart safely at 60:00** switch is on by default. It pauses, exits, and restarts as Joe at 60 minutes of game time to avoid training on the degraded long-run state you identified.

## Deliberate boundary

The script does **not** enter a name, click `SAVE`, or submit a leaderboard rank. It loops games and learns, but any public score submission remains a manual decision.

## Install and run

1. Install a userscript manager such as Violentmonkey or Tampermonkey.
2. Create a script, replace its contents with [`pine-autopilot.user.js`](./pine-autopilot.user.js), and save.
3. Open `https://pineandco.online/`. The enabled script arms itself after about one second and begins as Joe.
4. Add additional game windows if desired; they become experience workers automatically. Leave the windows active for the most consistent timing. Use **Stop** at any time; the central model is saved automatically.

Start v8 at **10× to 25×**. The bot polls at 10 ms and discards individual experience from any observed gap above 1.75 game seconds. A few timer warnings do not invalidate an evaluation by themselves: the accepted-run limit is a 10% dropped-decision rate, shown in the diagnostics. An external speed userscript controls the speed—the learner never changes it. Check its actual active multiplier with `window.__pineSpeed` in the Pine tab’s DevTools Console. If clean evaluations are frequently excluded or the largest time jump exceeds 1.75 seconds, reduce speed or tab count before collecting more data.

## Local tuning for this Mac

For the 12-core M4 Pro with 24 GB RAM, start at **three total tabs at 10×–25×** and only add a tab after tournament evaluations remain clean. Four tabs is a sensible practical ceiling at 25×. Train on power, keep Chrome hardware acceleration enabled, and in Chrome Performance settings add `pineandco.online` to **Always keep these sites active**. Keep Memory Saver enabled for unrelated tabs.

Do not use an approximate simulator for pretraining. A simulator with unvalidated collision or reward mechanics will add high-volume but misleading experience and degrade survival. The MPS trainer deliberately consumes recorded Pine transitions only, and the live tournament is its safety gate.

Run only one injected Pine bot in each game tab. Multiple tabs are supported; a second independent controller in the same tab will send competing inputs and corrupt the learning signal.
