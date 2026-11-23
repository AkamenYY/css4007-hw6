# HW6 submission

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation commands and exact run commands.

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

## Part B — FrozenLake

### Fixed policy, written before the experiments

State the action for each nonterminal state and your reasoning.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |

Link episode rows and example traces in outputs/. Explain slipping, variation across batches and why these runs do not train a policy.

## Part C — Custom solver

### Compare code with paper

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

