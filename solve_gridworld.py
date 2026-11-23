"""Part C: solve the paper map and the extended map, check the rules, simulate the policies.

    python solve_gridworld.py

Writes to outputs/:
  partC_paper_map.txt      V1, V2 check, final values, policy, convergence (paper map)
  partC_extended_map.txt   the same for the extended map (sand D, water W)
  rule_checks.txt          walls/borders, ice probabilities, floor rewards, terminal states
  partC_episodes.csv       one row per simulated episode
  partC_summary.md         the simulation table
"""
import csv
import random
import statistics
from pathlib import Path

from gridworld import ACTIONS, CONFIG, extended_map, paper_map
from value_iteration import best_actions, policy, q_value, solve, update

OUT = Path(__file__).resolve().parent / "outputs"
GAMMA = CONFIG["discount"]
SOLVER = CONFIG["solver"]
SIM = CONFIG["simulation"]


def fmt(x):
    return f"{x:.4f}".rstrip("0").rstrip(".")


def report(name, world):
    lines = [f"# {name}", "", "Map:", world.show(), ""]
    v0 = {s: 0.0 for s in world.states}
    v1 = update(world, v0, GAMMA)
    lines += ["V1 (one round from V0 = 0):", world.show(values=v1), ""]
    lines.append("V2 at the requested positions (Q of every action, using V1):")
    for r, c in CONFIG["paper_v2_positions"]:
        q = {a: q_value(world, v1, (r, c), a, GAMMA) for a in ACTIONS}
        best = max(q.values())
        ties = [a for a in ACTIONS if q[a] >= best - 1e-9]
        lines.append(f"  ({r},{c}) {world.letter((r, c))}: " + ", ".join(f"{a} {fmt(q[a])}" for a in ACTIONS)
                     + f"  -> V2 = {fmt(best)}, best: {', '.join(ties)}")
    result = solve(world, GAMMA, SOLVER["tolerance"], SOLVER["max_iterations"])
    values = result["values"]
    status = "converged" if result["converged"] else "DID NOT CONVERGE (safeguard reached)"
    lines += ["", f"Value iteration: {status} after {result['iterations']} iterations, "
                  f"final largest change = {result['final_change']:.3e} (tolerance {SOLVER['tolerance']})", "",
              "Final values:", world.show(values=values), "",
              "Best actions (all tied actions shown):", world.show(policy=best_actions(world, values, GAMMA)), "",
              "Tie-break rule: if two actions are equal (within 1e-9), take the first in the order up, right, down, left."]
    lines += ["", "Rounds until convergence (largest change per round):"]
    hist = result["history"]
    for k in range(1, len(hist)):
        change = max(abs(hist[k][s] - hist[k - 1][s]) for s in world.states)
        lines.append(f"  round {k:2d}: change {change:.3e}   value at start {hist[k][world.start]:.4f}")
    text = "\n".join(lines) + "\n"
    return text, result


def rule_checks():
    w, e = paper_map(), extended_map()
    lines = ["# Rule checks (run before trusting the solver)", ""]
    ok = True

    def check(label, condition, detail):
        nonlocal ok
        ok &= bool(condition)
        lines.append(f"[{'PASS' if condition else 'FAIL'}] {label}: {detail}")

    o = w.outcomes((1, 1), "up")
    check("border", o == [(1.0, (1, 1), -1)], f"(1,1) up -> {o} (stays, gets -1 of its own tile)")
    o = w.outcomes((1, 2), "down")
    check("wall", o == [(1.0, (1, 2), -3)], f"(1,2) mud, down into # -> {o} (stays, gets mud -3)")
    o = w.outcomes((3, 2), "up")
    check("wall", o == [(1.0, (3, 2), -1)], f"(3,2) up into # -> {o}")
    check("wall is not a state", (2, 2) not in w.states, f"states = {len(w.states)} tiles, (2,2) missing")
    for a in ACTIONS:
        o = w.outcomes((2, 3), a)
        check(f"ice sums to 1 ({a})", abs(sum(p for p, _, _ in o) - 1) < 1e-12,
              f"(2,3) {a} -> " + ", ".join(f"{p:.1f} to {n} reward {r}" for p, n, r in o))
    o = w.outcomes((2, 3), "left")
    check("ice slip resolved against wall", o[0] == (0.8, (2, 3), -1), f"(2,3) left -> {o}")
    check("mud reward", w.outcomes((1, 1), "right") == [(1.0, (1, 2), -3)], "entering M gives -3")
    check("sand reward", e.outcomes((3, 1), "right") == [(1.0, (3, 2), -2)], "entering D gives -2")
    check("water reward", e.outcomes((3, 3), "right") == [(1.0, (3, 4), -5)], "entering W gives -5")
    check("sand/water deterministic", e.outcomes((3, 2), "up") == [(1.0, (3, 2), -2)]
          and e.outcomes((3, 4), "left") == [(1.0, (3, 3), -1)], "moves from D and W have probability 1")
    check("goal reward", w.outcomes((1, 3), "right") == [(1.0, (1, 4), 10)], "entering G gives +10, no floor cost")
    check("hole reward", w.outcomes((3, 4), "up") == [(1.0, (2, 4), -10)], "entering H gives -10")
    res = solve(w, GAMMA, SOLVER["tolerance"], SOLVER["max_iterations"])
    check("no rewards after termination", res["values"][(1, 4)] == 0 and res["values"][(2, 4)] == 0,
          "V(G) = V(H) = 0 in every round; the simulator stops at G/H")
    lines += ["", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"]
    return "\n".join(lines) + "\n", ok


def run_episode(world, pol, seed):
    """One episode with a 100-step limit. The slip randomness comes from Random(seed)."""
    rng, s, rewards = random.Random(seed), world.start, []
    for _ in range(SIM["max_steps"]):
        s, reward = world.sample(s, pol[s], rng)
        rewards.append(reward)
        if world.is_terminal(s):
            outcome = "goal" if world.rule(s)["reward"] > 0 else "hole"
            break
    else:
        outcome = "time limit"
    discounted = sum(GAMMA ** t * r for t, r in enumerate(rewards))
    return {"steps": len(rewards), "total_reward": sum(rewards), "discounted_return": round(discounted, 6),
            "outcome": outcome}


def main():
    OUT.mkdir(exist_ok=True)
    checks, ok = rule_checks()
    (OUT / "rule_checks.txt").write_text(checks, encoding="utf-8")
    print(checks)

    rows, table = [], ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
                       "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, world, file in [("paper", paper_map(), "partC_paper_map.txt"),
                              ("extended", extended_map(), "partC_extended_map.txt")]:
        text, result = report(f"{name} map", world)
        pol = policy(world, result["values"], GAMMA)
        (OUT / file).write_text(text, encoding="utf-8")
        print(text)
        for batch in SIM["batch_seeds"]:
            eps = []
            for i in range(SIM["episodes_per_batch"]):
                seed = 1000 * batch + i
                ep = run_episode(world, pol, seed)
                eps.append(ep)
                rows.append({"map": name, "batch_seed": batch, "episode_seed": seed, **ep})
            count = lambda o: sum(e["outcome"] == o for e in eps)
            table.append(f"| {name} | {batch} | {count('goal')} | {count('hole')} | {count('time limit')} | "
                         f"{statistics.mean(e['steps'] for e in eps):.2f} | "
                         f"{statistics.mean(e['discounted_return'] for e in eps):.3f} |")
        table.append(f"| {name} | (value of start V(S)) | | | | | {result['values'][world.start]:.3f} |")

    with open(OUT / "partC_episodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "partC_summary.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))


if __name__ == "__main__":
    main()
