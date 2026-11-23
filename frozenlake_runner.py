"""Part B: random actions vs. my fixed policy on Gymnasium FrozenLake-v1 (4x4, default rewards).

    python frozenlake_runner.py

Writes to outputs/:
  partB_episodes.csv     one row per episode (600 rows)
  partB_summary.md       the results table
  partB_traces.txt       text-rendered episodes (one success, one failure per condition, if they happened)

Map (states are numbered row by row, 0..15):
    S  F  F  F       0  1  2  3
    F  H  F  H       4  5  6  7
    F  F  F  H       8  9 10 11
    H  F  F  G      12 13 14 15
"""
import csv
import random
import re
import statistics
from pathlib import Path

import gymnasium as gym

OUT = Path(__file__).resolve().parent / "outputs"
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3            # FrozenLake's action numbers
NAMES = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}
BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
MAX_STEPS = 100
GOAL, HOLES = 15, {5, 7, 11, 12}

# My fixed policy, written from the map BEFORE any run and never changed.
# Idea: the safe shortest path S(0) -> 4 -> 8 -> 9 -> 13 -> 14 -> G(15) goes down the left side
# and then right along the bottom, away from the holes 5, 7 and 11.
# Other states point back to that path or straight to the goal without touching a hole.
FIXED_POLICY = {
    0: DOWN,    # start: go down the left edge
    1: LEFT,    # back to the start (down would be hole 5)
    2: DOWN,    # to 6, then down to 10
    3: LEFT,    # away from hole 7
    4: DOWN,    # keep going down the left edge
    6: DOWN,    # to 10 (right would be hole 7)
    8: RIGHT,   # to 9 (down would be hole 12)
    9: DOWN,    # to 13 on the bottom row
    10: DOWN,   # to 14 (right would be hole 11)
    13: RIGHT,  # along the bottom row
    14: RIGHT,  # into the goal
}


def plain(frame):
    """Gymnasium paints the agent's tile red with terminal colour codes.
    For a text file we write that tile as a lowercase letter instead (s, f, h, g)."""
    return re.sub(r"\x1b\[41m(.)\x1b\[0m", lambda m: m.group(1).lower(), frame)


def run_episode(env, choose, seed, keep_frames=False):
    """reset() then step() until terminated or truncated, at most 100 steps."""
    state, _ = env.reset(seed=seed)
    frames = [plain(env.render())] if keep_frames else []
    steps, total, terminated, truncated = 0, 0.0, False, False
    actions = []
    while not (terminated or truncated) and steps < MAX_STEPS:
        action = choose(state)
        state, reward, terminated, truncated, _ = env.step(action)
        steps += 1
        total += reward
        actions.append(NAMES[action])
        if keep_frames:
            frames.append(plain(env.render()))
    if terminated:                       # terminal outcome wins, even on step 100
        outcome = "goal" if state == GOAL else "hole"
    else:
        outcome = "time limit"
    return {"steps": steps, "total_reward": total, "outcome": outcome, "final_state": state,
            "actions": actions, "frames": frames}


def main():
    OUT.mkdir(exist_ok=True)
    rows, traces = [], []
    table = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for slippery in (False, True):
        for policy_name in ("random", "fixed"):
            condition = f"{'slippery' if slippery else 'not slippery'} / {policy_name}"
            env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                           render_mode="ansi", max_episode_steps=MAX_STEPS)
            shown = set()
            for batch in BATCH_SEEDS:
                rng = random.Random(batch)          # seeded random action generator, one per batch
                if policy_name == "random":
                    choose = lambda s: rng.randrange(4)
                else:
                    choose = lambda s: FIXED_POLICY[s]
                episodes = []
                for i in range(EPISODES):
                    seed = 1000 * batch + i
                    ep = run_episode(env, choose, seed, keep_frames=True)
                    episodes.append(ep)
                    rows.append({"condition": condition, "slippery": slippery, "batch_seed": batch,
                                 "episode_seed": seed, "policy": policy_name, "steps": ep["steps"],
                                 "total_reward": ep["total_reward"], "outcome": ep["outcome"],
                                 "final_state": ep["final_state"]})
                    kind = "success" if ep["outcome"] == "goal" else "failure"
                    if kind not in shown:
                        shown.add(kind)
                        traces.append((condition, batch, seed, ep))
                n = lambda o: sum(e["outcome"] == o for e in episodes)
                table.append(f"| {'yes' if slippery else 'no'} | {policy_name} | {batch} | {n('goal')} | {n('hole')} | "
                             f"{n('time limit')} | {100 * n('goal') / EPISODES:.0f}% | "
                             f"{statistics.mean(e['steps'] for e in episodes):.2f} |")
            for kind in ("success", "failure"):
                if kind not in shown:
                    traces.append((condition, None, None, kind))
            env.close()

    with open(OUT / "partB_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "partB_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    lines = ["# FrozenLake traces (render_mode='ansi')",
             "The lowercase letter is where the agent is. The word in brackets is the last action.", ""]
    for condition, batch, seed, ep in traces:
        if batch is None:
            lines += [f"## {condition}: no {ep} happened in any of the 150 episodes", ""]
            continue
        lines += [f"## {condition}: {ep['outcome']} (batch {batch}, episode seed {seed}, {ep['steps']} steps)",
                  "actions: " + ", ".join(ep["actions"]), ""]
        frames = ep["frames"] if len(ep["frames"]) <= 12 else ep["frames"][:6] + ["...\n"] + ep["frames"][-5:]
        lines += [f.rstrip("\n") for f in frames] + [""]
    (OUT / "partB_traces.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
