"""Value iteration, written by hand (no library).

V(k+1, s) = max over a of  sum over s' of  P(s'|s,a) * (R + gamma * V(k, s'))

It only uses world.states, world.is_terminal and world.outcomes, so it works
for any map and any floor rules.
"""
from gridworld import ACTIONS


def q_value(world, values, s, action, gamma):
    """Expected value of doing `action` in state s, using the given values."""
    return sum(p * (reward + gamma * values[nxt]) for p, nxt, reward in world.outcomes(s, action))


def update(world, values, gamma):
    """One full round. Uses only the previous round's values. Terminal states stay 0."""
    return {s: 0.0 if world.is_terminal(s) else max(q_value(world, values, s, a, gamma) for a in ACTIONS)
            for s in world.states}


def solve(world, gamma, tolerance, max_iterations):
    """Repeat rounds until the biggest change is below `tolerance` (or the safeguard is hit)."""
    values = {s: 0.0 for s in world.states}
    history = [values]
    change = None
    for k in range(1, max_iterations + 1):
        new = update(world, values, gamma)
        change = max(abs(new[s] - values[s]) for s in world.states)
        values = new
        history.append(values)
        if change < tolerance:
            return {"values": values, "iterations": k, "final_change": change, "converged": True,
                    "history": history}
    return {"values": values, "iterations": max_iterations, "final_change": change, "converged": False,
            "history": history}


def best_actions(world, values, gamma, tie=1e-9):
    """All actions whose Q is within `tie` of the best one (ties), for every nonterminal state."""
    result = {}
    for s in world.states:
        if world.is_terminal(s):
            continue
        q = {a: q_value(world, values, s, a, gamma) for a in ACTIONS}
        top = max(q.values())
        result[s] = [a for a in ACTIONS if q[a] >= top - tie]
    return result


def policy(world, values, gamma):
    """One action per state. Tie-break: the first tied action in the order up, right, down, left."""
    return {s: acts[0] for s, acts in best_actions(world, values, gamma).items()}
