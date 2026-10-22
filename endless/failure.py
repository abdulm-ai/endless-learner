"""Learning from failure: death signatures and the death bank.

Every death is labelled by what killed the agent and what it was doing at the
time, for example "overhang/air" (jumped into an overhang) or "gap/ground"
(ran into a gap without jumping). The death bank keeps the game state from a
little before each death, and the trainer restarts some episodes from those
states so the agent practises the exact situation that killed it, the idea
behind "first return, then explore" (Ecoffet et al., 2021) and reverse
curriculum generation (Florensa et al., 2017).

An entry counts as fixed after the agent gets through it `fix_streak` times in
a row. The number of attempts that took is the "attempts to fix a failure" KPI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import CAUSES, POSTURES, State

PASS_MARGIN = 40  # ticks past the death tick that count as getting through


def signature(cause: int, posture: int) -> str:
    return f"{CAUSES[cause]}/{POSTURES[posture]}"


@dataclass
class DeathEntry:
    tier: int
    seed: int
    state: State
    death_tick: int
    signature: str
    episode: int
    attempts: int = 0
    passes: int = 0
    streak: int = 0


class DeathBank:
    def __init__(self, capacity: int = 1000, rewind: int = 20, fix_streak: int = 2, max_attempts: int = 30):
        self.capacity = capacity
        self.rewind = rewind  # decisions before the death to restart from
        self.fix_streak = fix_streak
        self.max_attempts = max_attempts
        self.entries: list[DeathEntry] = []
        self.fixed = 0
        self.gave_up = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, tier: int, seed: int, history: list[State], death_tick: int, sig: str, episode: int) -> bool:
        """Save the state `rewind` decisions before the death. Deaths right at the start are not worth a restart."""
        if len(history) < self.rewind:
            return False
        self.entries.append(DeathEntry(tier, seed, history[-self.rewind], death_tick, sig, episode))
        if len(self.entries) > self.capacity:
            self.entries.pop(0)
            self.evicted += 1
        return True

    def sample(self, rng: np.random.Generator) -> DeathEntry:
        return self.entries[int(rng.integers(len(self.entries)))]

    def report(self, entry: DeathEntry, passed: bool) -> str | None:
        """Record a restart. Returns "fixed" or "gave_up" when the entry leaves the bank."""
        entry.attempts += 1
        if passed:
            entry.passes += 1
            entry.streak += 1
        else:
            entry.streak = 0
        outcome = None
        if entry.streak >= self.fix_streak:
            outcome = "fixed"
            self.fixed += 1
        elif entry.attempts >= self.max_attempts:
            outcome = "gave_up"
            self.gave_up += 1
        if outcome:
            try:
                self.entries.remove(entry)
            except ValueError:
                pass
        return outcome

    def state(self) -> dict:
        return {
            "entries": [e.__dict__.copy() for e in self.entries],
            "fixed": self.fixed,
            "gave_up": self.gave_up,
            "evicted": self.evicted,
        }

    def load(self, s: dict) -> None:
        self.entries = [DeathEntry(**{**e, "state": tuple(e["state"])}) for e in s["entries"]]
        self.fixed, self.gave_up, self.evicted = s["fixed"], s["gave_up"], s["evicted"]
