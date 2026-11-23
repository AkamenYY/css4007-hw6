"""The Grid World from Parts A and C.

The floor rules (reward, deterministic or slippery, terminal, wall) are read from
data/maps.json. This file and the solver only ask the rules; they never name a floor.
So a new floor such as sand (D) or water (W) is added in maps.json, not in the code.

Positions are (row, column) and start at 1, like the paper map.
"""
import json
import random
from pathlib import Path

CONFIG = json.loads((Path(__file__).resolve().parent / "data" / "maps.json").read_text(encoding="utf-8"))

ACTIONS = ["up", "right", "down", "left"]          # also the tie-breaking order
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
SIDEWAYS = {"up": ("left", "right"), "down": ("left", "right"),
            "left": ("up", "down"), "right": ("up", "down")}
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}


class GridWorld:
    def __init__(self, rows, tiles=None):
        self.rows = rows
        self.tiles = tiles or CONFIG["tiles"]
        for letter in "".join(rows):
            if letter not in self.tiles:
                raise ValueError(f"no rule for floor '{letter}' in data/maps.json")
        # every tile that is not a wall is a state
        self.states = [(r, c) for r in range(1, len(rows) + 1) for c in range(1, len(rows[0]) + 1)
                       if self.rule((r, c)).get("traversable", True)]
        self.start = next(s for s in self.states if self.letter(s) == "S")

    def letter(self, s):
        return self.rows[s[0] - 1][s[1] - 1]

    def rule(self, s):
        return self.tiles[self.letter(s)]

    def is_terminal(self, s):
        return bool(self.rule(s).get("terminal"))

    def move(self, s, direction):
        """Where one move goes. Outside the map or into a wall: stay in place."""
        r, c = s[0] + MOVES[direction][0], s[1] + MOVES[direction][1]
        if not (1 <= r <= len(self.rows) and 1 <= c <= len(self.rows[0])):
            return s
        if not self.rule((r, c)).get("traversable", True):
            return s
        return (r, c)

    def directions(self, s, action):
        """Which way you really go, with probabilities. Depends only on the tile you leave."""
        rule = self.rule(s)
        if rule.get("movement") == "slippery":
            side = rule["perpendicular_each"]
            return [(rule["intended"], action), (side, SIDEWAYS[action][0]), (side, SIDEWAYS[action][1])]
        return [(1.0, action)]

    def outcomes(self, s, action):
        """The transition model: list of (probability, next state, reward).
        The reward is the reward of the tile where the move ends (also when blocked)."""
        result = {}
        for p, direction in self.directions(s, action):
            nxt = self.move(s, direction)
            result[nxt] = result.get(nxt, 0.0) + p
        return [(p, nxt, self.rule(nxt)["reward"]) for nxt, p in result.items()]

    def sample(self, s, action, rng: random.Random):
        """One random move for the simulator."""
        x, total = rng.random(), 0.0
        for p, nxt, reward in self.outcomes(s, action):
            total += p
            if x < total:
                return nxt, reward
        return nxt, reward  # rounding safety

    def show(self, values=None, policy=None, width=7):
        """Text map of the floor, the values or the policy arrows."""
        lines = []
        for r in range(1, len(self.rows) + 1):
            cells = []
            for c in range(1, len(self.rows[0]) + 1):
                s, letter = (r, c), self.letter((r, c))
                if letter == "#" or (s in self.states and self.is_terminal(s)):
                    cells.append(letter.center(width))
                elif policy is not None:
                    cells.append((letter + " " + "".join(ARROWS[a] for a in policy[s])).center(width))
                elif values is not None:
                    cells.append(f"{values[s]:{width}.3f}")
                else:
                    cells.append(letter.center(width))
            lines.append("|" + "|".join(cells) + "|")
        return "\n".join(lines)


def paper_map():
    return GridWorld(CONFIG["paper_map"])


def extended_map():
    return GridWorld(CONFIG["extended_map"])
