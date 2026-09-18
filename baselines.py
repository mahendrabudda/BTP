"""
baselines.py
============
Conventional traffic-signal strategies used to check whether the DQN
controller actually beats simple, non-learned alternatives (spec
section 11) — rather than just showing training reward went up.

All controllers share the interface:

    controller.reset()
    action = controller.select_action(env)   # env: TrafficEnv, for read-only inspection

Each returns an action in {0, 1} using the same "keep / switch" action
space as the DQN (see traffic_env.py), so they can be dropped into the
exact same environment for a fair, apples-to-apples comparison.
"""

import random


class FixedTimeController:
    """Switches phase on a fixed schedule, ignoring traffic state entirely."""

    def __init__(self, phase_duration: int = 10):
        self.phase_duration = phase_duration

    def reset(self) -> None:
        self._elapsed = 0

    def select_action(self, env) -> int:
        self._elapsed += 1
        if self._elapsed >= self.phase_duration:
            self._elapsed = 0
            return 1  # switch
        return 0  # keep


class RandomController:
    """Picks a uniformly random legal action each step."""

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def select_action(self, env) -> int:
        return self._rng.randrange(2)


class ActuatedController:
    """
    Demand-based heuristic: switches as soon as the minimum green has
    elapsed AND the waiting (red) pair's total queue exceeds the
    currently-green pair's, by a margin. Approximates real actuated
    signal controllers used in traffic engineering.
    """

    def __init__(self, margin: int = 2):
        self.margin = margin

    def reset(self) -> None:
        pass

    def select_action(self, env) -> int:
        if not env.is_switch_legal():
            return 0

        green_idx = env.PHASE_GROUPS[env.phase]
        red_idx = env.PHASE_GROUPS[1 - env.phase]
        green_demand = sum(env.queues[i] for i in green_idx)
        red_demand = sum(env.queues[i] for i in red_idx)

        return 1 if red_demand > green_demand + self.margin else 0
