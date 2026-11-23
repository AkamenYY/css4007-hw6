# HW6 submission

## AI-use declaration

I used **Claude (Anthropic)**:
- to check my Part A arithmetic after I wrote it on paper;
- to help write the code (`gridworld.py`, `value_iteration.py`, `solve_gridworld.py`, `frozenlake_runner.py`) and this report.

I ran every program myself. All numbers below come from the files in `outputs/`.

## Setup and reproduction

Tested with **Python 3.13**. Versions: `gymnasium==1.4.0`, `numpy==2.5.5` (pinned in `requirements.txt`). Part C uses only the Python standard library.

```bash
python3.13 -m venv .venv            # or: uv venv --python 3.13 .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python frozenlake_runner.py         # Part B -> outputs/partB_*
python solve_gridworld.py           # Part C -> outputs/partC_*, outputs/rule_checks.txt
```

No API key is needed. Every run uses fixed seeds, so running the commands again gives the same files.

| File | What it is |
| --- | --- |
| `partA/` | my Part A paper calculations (photo) |
| `gridworld.py` | the map, the move rules and the transition model; floor rules are read from `data/maps.json` |
| `value_iteration.py` | value iteration written by hand, and policy extraction |
| `solve_gridworld.py` | Part C: V1/V2 check, solve both maps, rule checks, simulation |
| `frozenlake_runner.py` | Part B: Gymnasium FrozenLake experiments |

---

## Part A — Paper calculations

Photos of my handwritten work (in Russian): [page 1](partA/partA_page1.jpg) (map, V1) and [page 2](partA/partA_page2.jpg) (V2 at the three positions). They were committed before any program.

Two writing slips on the photos, corrected here: on page 1 the last ice term should be `0.1(-1)`, not `0.1(-0.1)`; on page 2 the ice "right" line should end with `0.1(-1.9)`, not `0.9(-1.9)`. The results written on the photos (-1, -7.39) are correct. Below is the full typed version.

**1. The problem in my own words**
- **States:** the 9 tiles where the agent can stand and act: (1,1), (1,2) M, (1,3), (2,1), (2,3) I, (3,1) S, (3,2), (3,3), (3,4). G (1,4) and H (2,4) are terminal states with value 0. The wall (2,2) is not a state.
- **Actions:** up, right, down, left.
- **Transitions:** on S, `.` and M the move always happens. On ice the chosen direction happens with 0.8, each sideways direction with 0.1. A move into a wall or off the map leaves the agent in place.
- **Rewards:** the reward of the tile where the move ends: -1 floor, -3 mud, +10 goal, -10 hole. A blocked move still pays the reward of the tile you stay on.
- **Goal:** maximize the **expected discounted sum of rewards**, `r0 + 0.9 r1 + 0.9² r2 + ...`: reach G with as little cost and risk as possible.

**2. V1** (V0 = 0, so `V1 = best expected reward of one move`)

```text
| -1 | -1 | 10 | G |
| -1 | #  | -1 | H |
| -1 | -1 | -1 | -1|
```

Only (1,3) is different: moving right enters G (+10). Everywhere else the best one-step reward is -1. Mud (1,2): right or left gives -1, up/down are blocked and pay mud -3. Ice (2,3): left = 0.8×(-1, blocked) + 0.1×(-1) + 0.1×(-1) = **-1**; up = 0.8(-1) + 0.1(-1) + 0.1(-10) = -1.9.

**3. V2 using only V1** (`Q = sum of P × (R + 0.9 × V1(next))`)

(1,2) Mud:
- up: blocked, stay in M: -3 + 0.9(-1) = **-3.9**
- right: to (1,3): -1 + 0.9(10) = **8**
- down: wall, stay in M: -3 + 0.9(-1) = **-3.9**
- left: to (1,1): -1 + 0.9(-1) = **-1.9**
- **V2 = 8 (right)**

(1,3):
- up: blocked, stay: -1 + 0.9(10) = **8**
- right: to G: 10 + 0.9(0) = **10**
- down: to ice: -1 + 0.9(-1) = **-1.9**
- left: to mud: -3 + 0.9(-1) = **-3.9**
- **V2 = 10 (right)**

(2,3) Ice:
- up: 0.8(-1 + 0.9·10) + 0.1(-1 + 0.9·(-1)) [left: wall, stay] + 0.1(-10 + 0) [right: H] = 6.4 - 0.19 - 1 = **5.21**
- right: 0.8(-10) + 0.1(-1 + 0.9·10) [up] + 0.1(-1 + 0.9·(-1)) [down] = -8 + 0.8 - 0.19 = **-7.39**
- down: 0.8(-1 + 0.9·(-1)) + 0.1(-1.9) [left: stay] + 0.1(-10) [right: H] = -1.52 - 0.19 - 1 = **-2.71**
- left: 0.8(-1.9) [wall, stay] + 0.1(8) [up] + 0.1(-1.9) [down] = -1.52 + 0.8 - 0.19 = **-0.91**
- **V2 = 5.21 (up)**

**4. Best actions after two rounds:** (1,2) **→**, (1,3) **→**, (2,3) **↑**. There are no ties at these three positions.

Two rounds do not prove the policy is optimal. V2 only looks **two moves ahead**. Values far from the goal (for example the start) are still -1 or -1.9 and keep changing in later rounds. A better action can appear once the +10 travels further back through the map. Only when the values stop changing (convergence) can we trust the policy for all states.

---

## Part B — FrozenLake

`FrozenLake-v1`, `map_name="4x4"`, default rewards (+1 for the goal, 0 otherwise), default slip rule. Code: `frozenlake_runner.py`.

```text
S  F  F  F       0  1  2  3
F  H  F  H       4  5  6  7
F  F  F  H       8  9 10 11
H  F  F  G      12 13 14 15
```

### Fixed policy, written before the experiments

| State | Action | Reason |
| --- | --- | --- |
| 0 (S) | down | go down the left edge |
| 1 | left | back to the start (down is hole 5) |
| 2 | down | to 6, then 10 |
| 3 | left | away from hole 7 |
| 4 | down | continue down the left edge |
| 6 | down | to 10 (right is hole 7) |
| 8 | right | to 9 (down is hole 12) |
| 9 | down | to 13, the bottom row |
| 10 | down | to 14 (right is hole 11) |
| 13 | right | along the bottom row |
| 14 | right | into the goal |

The idea is the **shortest safe path** S → 4 → 8 → 9 → 13 → 14 → G (6 steps). It is stored in `FIXED_POLICY` and was not changed after any run.

The runner uses `reset(seed=...)` and `step()`, stops on `terminated` or `truncated`, with a 100-step limit. Episode `i` of batch `b` uses seed `1000·b + i`. The random policy uses `random.Random(batch_seed)`. The outcome comes from the final state: 15 = goal, a terminal hole = hole, truncated = time limit.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 2 | 48 | 0 | 4% | 9.78 |
| no | random | 22 | 0 | 50 | 0 | 0% | 6.90 |
| no | random | 33 | 0 | 50 | 0 | 0% | 7.44 |
| no | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0% | 8.44 |
| yes | random | 22 | 1 | 49 | 0 | 2% | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0% | 7.30 |
| yes | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| yes | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| yes | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Files: [episode rows](outputs/partB_episodes.csv) (600 rows), [table](outputs/partB_summary.md), [text-rendered traces](outputs/partB_traces.txt). The traces file has one success and one failure for every condition that had them. The only missing case: **"not slippery / fixed" never failed** (150/150 goals), so there is no failed trace for it.

**What slipping changed.** Without slipping, my fixed policy reached the goal in **all 150 episodes in exactly 6 steps**. With `is_slippery=True`, FrozenLake moves in the chosen direction only **1/3** of the time and slides to each side 1/3 of the time. My path goes right next to holes, for example "down" from state 4 can slide right into hole 5. That happened in 113 of the 146 failures. Success fell to **4/150 (2.7%)**, barely better than random (1/150). The policy was designed for deterministic moves; a slip-aware policy would instead push *away* from the holes. I kept the policy unchanged, as the task requires.

**How consistent the batches were.** In deterministic conditions the fixed policy gave identical batches. The random policy varied (2, 0, 0 goals). With slipping, batches gave 1, 2 and 1 goals. With only 50 episodes and so few successes, the batch-to-batch differences are noise.

**Why one success is not enough.** The random policy reached the goal 2 times in batch 11 and 0 times in batches 22 and 33. One lucky episode says nothing about how often a policy succeeds. You need many episodes and several seeds to estimate a rate.

**Why this is not training.** Both policies are fixed before the runs. No reward is used to change any action, so after 600 episodes the policies are exactly the same as before. We only **measure** policies. Training (learning) would update the policy or values from the rewards it gets.

---

## Part C — Custom solver

Code: `gridworld.py` builds the transition model from the map and the floor rules in `data/maps.json`. `value_iteration.py` is generic: it only asks for states, terminal flags and outcomes. Adding sand and water needed **no change to the solver**, only the rules in `maps.json` and a new map.

### Compare code with paper

Printed by `python solve_gridworld.py` ([outputs/partC_paper_map.txt](outputs/partC_paper_map.txt)):

```text
V1:
| -1.000| -1.000| 10.000|   G   |
| -1.000|   #   | -1.000|   H   |
| -1.000| -1.000| -1.000| -1.000|

(1,2) M: up -3.9, right 8, down -3.9, left -1.9  -> V2 = 8, best: right
(1,3) .: up 8, right 10, down -1.9, left -3.9  -> V2 = 10, best: right
(2,3) I: up 5.21, right -7.39, down -2.71, left -0.91  -> V2 = 5.21, best: up
```

**The code matches my paper work exactly**, so there were no discrepancies to correct.

### Values, policies and convergence

Files: [paper map](outputs/partC_paper_map.txt), [extended map](outputs/partC_extended_map.txt).

| Map | Iterations | Final largest change | Converged? |
| --- | --- | --- | --- |
| paper `.M.G / .#IH / S...` | 12 | 2.17e-07 | yes (< 1e-6) |
| extended `.M.G / .#IH / SD.W` | 11 | 2.41e-07 | yes (< 1e-6) |

```text
Paper map values                        Paper map policy
|  4.200|  8.000| 10.000|   G   |       |  . →  |  M →  |  . →  |   G   |
|  2.780|   #   |  5.824|   H   |       |  . ↑  |   #   |  I ↑  |   H   |
|  1.536|  2.818|  4.242|  2.818|       |  S →  |  . →  |  . ↑  |  . ←  |

Extended map values                     Extended map policy
|  4.200|  8.000| 10.000|   G   |       |  . →  |  M →  |  . →  |   G   |
|  2.780|   #   |  5.824|   H   |       |  . ↑  |   #   |  I ↑  |   H   |
|  1.502|  2.818|  4.242|  2.818|       |  S ↑  |  D →  |  . ↑  |  W ←  |
```

The policy is chosen by comparing the four Q-values with the final values. **Tie-break:** if two actions are equal within 1e-9, I take the first in the order up, right, down, left. In these maps no state had a tie.

### Simulation results

Three batches of 50 episodes, seeds `1000·b + i`, 100-step limit. The slip randomness is `random.Random(episode_seed)`. Discounted return = `r0 + 0.9 r1 + 0.9² r2 + ...`. Rows: [outputs/partC_episodes.csv](outputs/partC_episodes.csv).

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 42 | 8 | 0 | 4.98 | 0.891 |
| paper | 22 | 46 | 4 | 0 | 5.04 | 1.915 |
| paper | 33 | 48 | 2 | 0 | 5.04 | 2.492 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

The computed value of the start is **V(S) = 1.536** (paper) and **1.502** (extended).

### Explain your results

**Only the start changed.** All values except S are the same on both maps, and so are all actions except S.
- On the **paper map** the start goes **right** (S → . → . → ice ↑ → . → G). It is the cheapest route (only -1 tiles), but it crosses the ice. From the ice, "up" slides right into the hole with probability 0.1. Its value is Q(S, right) = -1 + 0.9 × 2.818 = **1.536**.
- The safe route goes **up** (S → . → . → mud → . → G). It pays -3 for the mud but never touches ice: Q(S, up) = -1 + 0.9 × 2.780 = **1.502**.
- The ice route is only slightly better (1.536 vs 1.502). On the **extended map** the first step right now enters **sand (-2)**, so Q(S, right) = -2 + 0.9 × 2.818 = **0.536**, and the policy switches to the safe mud route. One extra point of cost was enough to flip the decision.
- **Water** at (3,4) changed nothing: the value of standing on W is still 2.818 (moves from W are deterministic, and its best action is left). No good path enters W, so its -5 entry cost is never paid.

**How the policy handled ice.** On ice it always chooses **up**, towards the goal, even though it slips right into the hole 10% of the time. The other actions are worse (left -0.91, down -2.71, right -7.39 after two rounds). If it slips left, it hits the wall and stays on the ice, then tries again. So from ice the chance of finally falling in is 0.1 / 0.9 ≈ **11%**. In the simulation the paper map lost **14/150 (9.3%)** episodes in the hole, close to this. The extended map avoids the ice, so it won all 150 episodes.

**Simulations vs. computed values.** V(S) is an **expected** value over infinitely many episodes with no step limit. A simulation is a **finite sample**:
- On the extended map everything is deterministic, so every episode returns exactly **1.502** = V(S).
- On the paper map each episode returns one of only three numbers: **3.122** (goal in 5 steps, 121 episodes), **1.810** (goal after one slip, 15 episodes) or **-10.0** (hole, 14 episodes). Batch means were 0.891, 1.915 and 2.492, and the mean of all 150 is **1.766**, near V(S) = 1.536. The spread shows that 50 episodes is a small sample: one hole instead of a goal moves a batch mean by (3.122 + 10) / 50 ≈ 0.26.
- The 100-step limit cuts off the infinite horizon, but no episode came close to it (the longest took 6 steps), so here it did not matter.

### Rule checks

Saved in [outputs/rule_checks.txt](outputs/rule_checks.txt); all 16 checks pass. Examples:

```text
[PASS] border: (1,1) up -> [(1.0, (1, 1), -1)] (stays, gets -1 of its own tile)
[PASS] wall: (1,2) mud, down into # -> [(1.0, (1, 2), -3)] (stays, gets mud -3)
[PASS] ice sums to 1 (up): (2,3) up -> 0.8 to (1, 3) reward -1, 0.1 to (2, 3) reward -1, 0.1 to (2, 4) reward -10
[PASS] mud reward: entering M gives -3
[PASS] sand reward: entering D gives -2
[PASS] water reward: entering W gives -5
[PASS] goal reward: entering G gives +10, no floor cost
[PASS] no rewards after termination: V(G) = V(H) = 0 in every round; the simulator stops at G/H
```

**Note:** D and W are custom floor letters for this exercise, not standard FrozenLake tiles. That is why Part C uses my own simulator, not Gymnasium.
