# endless-learner

A small game and an agent that learns it from its own failures, training for days on a laptop CPU. Every death becomes training data, the game gets harder as the agent gets better, and a live page shows it learning. The game, the neural network, backpropagation, the optimiser and the reinforcement learning are written from scratch on NumPy: no PyTorch, no Gym, no pretrained weights.

![Ghost race: the agent after 30 minutes of training (purple) against 29.5 hours (blue), on a tier-6 level it has never seen](results/race.gif)

*The same unseen tier-6 level, played by the agent after 30 minutes of training (purple, dies under an overhang) and after 29.5 hours (blue, finishes). [results/learning.mp4](results/learning.mp4) shows the same level at seven ages, from untrained to 29.5 hours.*

The short version of what happened, over about 218 active hours of training across 19 runs: the agent climbed from tier 0 to tier 7 in its first 6.3M steps, then spent the next 220M steps between tiers 5 and 7 without getting past 7. At tiers 6 and 7 it beats a hand-written player that reads the level directly. Of the mechanisms built to make it learn from failure, the one that mattered most was the plainest (replaying old tiers), and two of them made it worse. Continual backprop kept the network healthy for the whole run, but that only sometimes turned into better play. Details and every number are in [Results](#results).

## How it works

```mermaid
flowchart LR
  G["Level generator<br/>seed + tier<br/>solvability check"] --> E["Game<br/>headless, deterministic"]
  E -->|"9 rays, terrain scan,<br/>5 body features"| A["Agent<br/>NumPy MLP, Double DQN<br/>acts with averaged weights"]
  A -->|"noop / short / long / duck"| E
  E -->|n-step transitions| R[("Replay<br/>prioritised by TD error<br/>+ reservoir of all history")]
  E -->|"death: state 20 decisions before"| D[("Death bank")]
  D -->|restart episodes there| E
  R --> T["Trainer<br/>backprop + Adam by hand<br/>target network"]
  T --> A
  C["Curriculum<br/>exams promote the tier<br/>prioritised level replay"] --> G
  V["Eval<br/>fixed held-out seeds, every tier"] --> M[("metrics.sqlite")]
  T -.-> M
  M --> W["Live view<br/>game, charts, ghost races"]
```

**The game.** A side-scrolling runner. Each level is a pure function of a tier and a seed: low and tall blocks, gaps, overhangs to duck under, blocks that slide towards the agent, and from tier 5 on, tight combinations of them. The agent can do nothing, jump short, jump long or duck. Tiers have no top: speed, density and the obstacle mix keep turning up until they reach caps set by the physics. Every level is checked solvable before it is handed out, by a depth-first search over jump timings that runs on the same physics code as the game, so no death is the level's fault.

**What the agent sees.** Nine rays fan out from the front of its body. Each reports the distance to the first hit, what was hit (ground, pit, block, overhang) and how fast that distance is closing, which is how a moving block shows up. A terrain scan samples the course every half unit for 12 units ahead: ground or pit, block height, overhang above, moving-block speed. Five more numbers describe the agent: height, vertical speed, on the ground, ducking, scroll speed. That is 155 inputs and no pixels, so a 155-128-128-4 MLP is enough and it trains on a CPU.

**Acting.** Double DQN (van Hasselt et al., 2016) with 3-step returns, a hard target sync every 2,000 updates and Adam at 1e-4. The agent acts, and is evaluated and saved, with an exponential moving average of the online weights (Polyak averaging, 0.5% per update), not the online weights themselves. See "What didn't work" for why.

**Learning from failure.**

- *Prioritised replay* (Schaul et al., 2016). Transitions are replayed in proportion to their last TD error, with importance weights to correct the bias. The run logs |TD| on transitions just before a death against routine ones, which shows whether replay focuses where it should.
- *The death bank.* Every death saves the game state from 20 decisions earlier. A quarter of episodes restart from one of those states, so the agent practises the exact situation that killed it (the idea of "first return, then explore", Ecoffet et al., 2021, and reverse curriculum generation, Florensa et al., 2017). An entry counts as fixed after two passes in a row; the attempts that took are logged.
- *Death signatures.* Each death is labelled by cause and posture: `overhang/air` is jumping into an overhang, `gap/ground` is running into a gap.

**Never stopping.**

- *Exams drive the curriculum.* 30% of fresh levels at the current tier are played with no exploration, and only those count for promotion: with 3% random actions per decision, exploring episodes die often enough to hide how good the policy is. Promotion at 80% over 50 exams. Demotion only after a collapse: the tier went well once (50% or better over a window) and then fell under 20%.
- *Prioritised level replay* (Jiang et al., 2021) inside a tier: seeds whose last episode had a high mean TD error come back more often.
- *Against forgetting:* 15% of episodes replay a passed tier, and a reservoir sample over the whole history (25% of every batch) keeps old transitions in training after the ring buffer has overwritten them.
- *Against plasticity loss:* the run tracks dead units, feature rank (srank, Kumar et al., 2021) and weight norm. `--cbp` turns on continual backprop (Dohare et al., 2024), which replaces a trickle of the least useful hidden units with fresh ones.

**Running forever.** One process with NumPy capped to one thread. Buffers are fixed in size, but memory still creeps up by about 4 MB per active hour (cause not yet tracked down), so each run stops itself cleanly with a checkpoint if it passes 1.5 GB. A checkpoint of everything (weights, optimiser, replay, death bank, curriculum, RNG state) every 5 minutes, written atomically; starting `train` on the same folder resumes exactly, and metrics logged after the last checkpoint are dropped and counted as lost. Time is counted in active hours, with gaps capped, so a sleeping laptop doesn't count. A hall-of-fame snapshot of the weights every active hour.

## Decisions and trade-offs

- **Double DQN, not PPO.** Learning from failure means going back to old experience. Off-policy replay does that; on-policy methods use each transition once and throw it away.
- **Rays and a scan, not pixels.** Pixels would need convolutions and far more compute. Rays and the scan carry what matters (distance, type, height, closing speed) and work on any generated level.
- **NumPy, not a framework.** Every gradient is written out and checked against central differences (`python -m endless gradcheck`, max relative error around 1e-8). The cost is speed: a median of 870 to 1,180 environment steps per second on one core of an i7-1265U over the long runs (with up to five runs sharing the laptop), with a learning update every 4 steps, and 240 to 500 MB of RAM per run, most of it the 100k-transition replay buffer.
- **Two ticks per decision.** Halves the number of decisions per level, which makes credit assignment easier, at no cost in control: jumps last 10 to 15 decisions.
- **Promotion on exams, not training episodes.** See above. The held-out evaluation seeds never drive training: they are only measured.
- **A solvable-level guarantee.** Generation retries until the solver finds a plan, which gets slow at high tiers (tens to hundreds of milliseconds per level beyond tier 15). Levels are cached and replayed seeds are common, so it stays a small share of the time.

## What didn't work

Each of these was found with short runs on this laptop (several in parallel, one core each). The numbers are from those runs, not from the long run.

- **Rays alone.** A fixed fan of rays touches the ground at only four distances (0.8, 1.6, 3.1 and 8.2 units ahead), so the agent couldn't tell where a gap's edge was between them. In four runs of 1.2M steps each, none got past tier 1, the first tier with gaps. Landing in a gap or walking into one killed 26 to 44% of exams. With the terrain scan added, four runs all passed tier 1 within 230k to 340k steps.
- **Acting with the online weights.** Every update flips the greedy action in some states (policy churn), and an episode is played across about a hundred updates. In one tier-0 run, frozen snapshots of the network passed 100% of 100 random levels while the exams played by the moving network passed about 55%. With the scan in place and 1.5M steps per run, acting with averaged weights reached frontier tier 4 on both seeds. Acting with the online weights reached tier 2 and tier 4. That is two seeds each: a direction, not a measurement.
- **Learning rate 3e-4.** On tier 0 alone, held-out success at 3e-4 went 1.0, 1.0, 0.6, 0.45 across consecutive eval rounds. At 1e-4 it held 1.0 for four rounds before dipping. Both still wobble, which is part of what the averaged weights are for.
- **Promoting on training episodes.** With 3% random actions per decision, exploring episodes passed about 40% of tier-0 levels while the same network played greedily passed 100% of the held-out ones. The curriculum never moved, which is why exams exist.
- **Demoting on first contact.** A new tier is hard at first. With demotion at "under 20% over 50 exams", runs bounced between tiers 2 and 3 one to four times in 1.5M steps, each round trip costing about 230k to 270k steps.

## Results

Cut-off: 2026-10-25. Everything below comes from `python scripts/results.py`, which reads the runs' `metrics.sqlite` files up to fixed step counts and writes [results/results.json](results/results.json) and the charts. Method: every 200k steps, the agent plays 20 held-out levels per tier with no exploration. Those seeds are never trained on. Single eval rounds are noisy, so each number is an average over a window of rounds.

| Run | What | Steps | Active hours |
|-----|------|-------|--------------|
| `runs/main` | full config, seed 0 | 226M | 76.0 |
| `runs/cbp` | full config plus continual backprop, seed 0 | 226M | 59.0 |
| `runs/plain-s1`, `runs/cbp-s1` | the same pair on seed 1 | 100M each | 31.7, 29.7 |
| `runs/abl/*` | 5 variants, seeds 1 to 3 | 5M each | 1.3 to 1.6 each |

### The climb, then the plateau

![Training tier over the seed-0 runs](results/climb.png)

The plain seed-0 run reached tiers 1 to 7 at 0.09M, 0.25M, 0.42M, 0.60M, 0.75M, 3.3M and 6.3M steps (2.4 active hours for all seven). It spent the remaining 220M steps between tiers 5 and 7: three demotions, two climbs back, and never past 7. All seven promotions came in the first 3% of the run. No run held tier 8: the plain seed-1 run was promoted at 19.8M steps and demoted at 34.2M.

Against the baselines on the same held-out levels: a random player never finishes a level at any tier. The rule-based player reads the level directly and times every jump to centre its arc over the next obstacle. It finishes every level up to tier 4, then 0.60 at tier 5 and 0.30 at tiers 6 and 7. Averaged over the long runs, the learned agent gets 0.52 to 0.65 at tier 6 (from 50M to 100M steps) and 0.49 to 0.63 at tier 7.

### Ablations: removing one piece at a time

![Ablation results](results/ablations.png)

| Variant, 5M steps | Held-out success, tiers 0 to 5 | Range over 3 seeds | Seeds that reached tier 6 / 7 |
|---|---|---|---|
| full config | 0.76 | 0.74 to 0.79 | 2 / 0 |
| no prioritised replay | 0.88 | 0.86 to 0.92 | 3 / 3 |
| no death-bank restarts | 0.88 | 0.84 to 0.90 | 3 / 0 |
| no curriculum (always tier 6) | 0.89 | 0.83 to 0.91 | no promotions by design |
| no old-tier replay (no reservoir, no passed-tier episodes) | 0.21 | 0.14 to 0.29 | 3 / 3 |

Held-out success is averaged over the eval rounds from 4M to 5M steps.

- **Replaying old tiers is what holds it together.** Without it, every seed climbed fastest (tier 7 by 1.9M to 3.3M steps) and forgot the easy tiers almost completely: averaged over the three seeds, 0.05 to 0.23 on tiers 0 to 3.

  ![Held-out success by tier, full config against no old-tier replay](results/forgetting.png)

- **Prioritised replay, the death bank and the curriculum each made it worse at this budget.** Removing any one of them beat the full config on every seed, and the seed ranges don't overlap. I think the reason is that each one narrows what the agent trains on. Prioritised replay favours surprising transitions, the death bank spends a quarter of episodes on fragments just before a death, and the curriculum keeps the agent on one tier. Together that's a lot of skew, but I haven't tested that explanation. It also only holds at 5M steps, and a longer budget could change it.
- **The death bank did its local job well while not helping overall.** In the main run it restarted 323,444 times and fixed 23,117 failures, in a median of 2 attempts, and never gave up on one. Transitions just before a death carried 5.5x the TD error of routine ones (0.56 against 0.10), so prioritised replay was focusing where it was meant to. None of that showed up as a better agent.

### Plasticity over 226M steps

![Dead units and feature rank over training](results/plasticity.png)

The plain network loses plasticity steadily, as Dohare et al. (2024) describe. At 226M steps, 21% of its hidden units never fire, the feature rank (srank, out of 128) is 58, and the weight norm is 194. With continual backprop the same config holds dead units at 0.4% and srank at 116, with a weight norm of 148. Both seeds show the same picture.

![Tier-7 held-out success, plain against continual backprop](results/tier7.png)

Whether that matters for play depends on the seed:

- **Seed 0:** the plain run had two long dips on tier 7. It fell from 0.59 to 0.17 between 40M and 100M steps and from 0.59 to 0.36 after 200M. After its first 10M steps, the continual-backprop run stayed between 0.50 and 0.65 the whole time. In between, from 100M to 200M, the two were level.
- **Seed 1:** the plain run never dipped. It averaged 0.63 on tier 7 to 100M steps, slightly above continual backprop's 0.58.
- **Easier tiers:** on tiers 0 to 6, all four runs averaged 0.74 to 0.76.

So continual backprop removes the loss of plasticity completely, and in my reading it protects against the dips, but on this evidence the dips are not guaranteed to happen. Neither variant got past tier 7, so whatever stops the agent there is not plasticity loss alone. My guess is capacity or the observation, but I haven't tested either.

### Systems

The main run went 226M steps and 1.29M episodes over 76 active hours, with 0 NaN recoveries and 0 resumes. The continual-backprop run resumed once when its step limit was raised, and lost 0 steps. A full checkpoint is 159 MB and takes 0.2 s to write. RAM went from 238 MB to 501 MB over the main run, about 3.7 MB per active hour.

## Run it

Python 3.10+ and NumPy.

```
pip install numpy
python -m endless train --run runs/main
```

Open http://127.0.0.1:8765 for the live view: the agent playing a held-out level with its rays drawn, the charts above, and ghost races between any two hall-of-fame snapshots on the same level.

| Command | What it does |
|---------|--------------|
| `python -m endless train --run runs/main` | Train, or resume if the folder has a checkpoint. `--max-hours`, `--max-steps` to stop on their own; `--max-rss-mb` (default 1500) stops cleanly with a checkpoint if the process ever uses more RAM than that. |
| `python -m endless stop --run runs/main` | Stop cleanly from another terminal: checkpoint, then exit. Ctrl+C does the same. |
| `python -m endless view --run runs/main` | The live view on its own, for a run that is training in another process or has finished. |
| `python -m endless report --run runs/main` | Every KPI. `--hof` also races the newest snapshot against each older one. |
| `python -m endless baseline` | Random and rule-based players on the held-out levels. |
| `python -m endless eval --run runs/main --net h0012` | Any snapshot on the held-out levels. |
| `python -m endless gradcheck` | Backprop against numerical gradients. |
| `python scripts/results.py`, `python scripts/media.py` | Rebuild the results, charts, videos and GIF in `results/` from the runs. Charts need matplotlib, and videos need Pillow and ffmpeg; nothing else does. |
| `python -m unittest discover -s tests -t .` | Tests: physics, solver plans replayed in the environment, rays, sum tree, n-step returns, checkpoint and exact resume. |

`python scripts/ablations.py` runs the ablations below at a fixed step budget (5M) on seeds 1 to 3, three at a time, waiting for free RAM before each start and resuming anything unfinished. `--after runs/cbp` makes it wait for another run to finish first. Ablations are flags on `train`: `--no-per`, `--no-restarts`, `--no-curriculum --fixed-tier 6`, `--no-reservoir`, `--no-old-tiers`, `--no-plr`, and `--cbp` to add continual backprop. Use a separate `--run` folder for each.

Keeping it running is up to you: plug the laptop in, stop it from sleeping, and start `train` again after a reboot. It carries on where it stopped.

## Limitations

- One game, and a small one. Rays make the problem much easier than learning from pixels.
- Single-threaded NumPy is slow next to a GPU framework. The point is that every piece is visible, not speed.
- "Forever" is bounded by the physics: past some tier the generator can't make harder solvable levels, and the knobs sit at their caps.
- Death signatures are coarse on purpose (cause and posture), so they count types of failure, not individual situations. The death bank is where individual situations are tracked.
- The ablations are 3 seeds at 5M steps, and the continual-backprop comparison is 2 seeds. They are good enough to show a direction, not to measure small effects.
- Memory creeps up by about 4 MB per active hour, and I haven't found the cause. The 1.5 GB cap stops a run cleanly long before it matters on this laptop.
- The agent plateaus at tier 7. I don't know yet whether the network, the observation or the training setup is the limit.

## References

Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015. van Hasselt, Guez and Silver, "Deep Reinforcement Learning with Double Q-learning", AAAI 2016. Schaul et al., "Prioritized Experience Replay", ICLR 2016. Ecoffet et al., "First return, then explore", Nature 2021. Florensa et al., "Reverse Curriculum Generation for Reinforcement Learning", CoRL 2017. Jiang, Grefenstette and Rocktäschel, "Prioritized Level Replay", ICML 2021. Cobbe et al., "Leveraging Procedural Generation to Benchmark Reinforcement Learning", ICML 2020. Dohare et al., "Loss of plasticity in deep continual learning", Nature 2024. Lyle et al., "Understanding plasticity in neural networks", ICML 2023. Kumar et al., "Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning", ICLR 2021. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017. Chaudhry et al., "Riemannian Walk for Incremental Learning", ECCV 2018. Lopez-Paz and Ranzato, "Gradient Episodic Memory for Continual Learning", NeurIPS 2017. Kingma and Ba, "Adam", ICLR 2015.

## License

MIT
