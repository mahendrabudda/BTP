"""
traffic_env.py
===============
Gym-style TrafficEnv for a single 4-way intersection.

──────────────────────────────────────────────────────────────────────────
STATE  (10-dim, all normalized to [0, 1])
──────────────────────────────────────────────────────────────────────────
    [ q_right, q_down, q_left, q_up,          # queue length / max_queue
      w_right, w_down, w_left, w_up,          # steps waited  / max_wait_norm
      current_phase,                          # 0.0 = horizontal green, 1.0 = vertical green
      phase_elapsed ]                         # steps in current phase / min_green_steps, clipped to 1

    Queue counts alone (the previous design) cannot tell the agent *how
    long* a lane has been starved — two states with identical queues can
    require opposite actions depending on wait history. Including wait
    time and phase/phase-elapsed lets the DQN learn starvation-avoidance
    itself, instead of needing an external hard-coded override.

──────────────────────────────────────────────────────────────────────────
ACTION  (discrete, 2 actions)
──────────────────────────────────────────────────────────────────────────
    0 → keep current phase
    1 → switch to the other phase

    Phase 0 = green for {right, left} (horizontal pair)
    Phase 1 = green for {down, up}    (vertical pair)

    Safety: a "switch" requested before `min_green_steps` have elapsed in
    the current phase is masked to a no-op by the environment itself, so
    illegal/unsafe transitions (e.g. flashing green for <1s) are
    structurally impossible rather than merely discouraged.

──────────────────────────────────────────────────────────────────────────
REWARD
──────────────────────────────────────────────────────────────────────────
    reward = (total_wait_before_step - total_wait_after_step) - switch_cost

    This is the direct "reduce cumulative waiting time" formulation.
    It is mathematically justified because the sum of per-step rewards
    over an episode telescopes to:

        sum(r_t) = total_wait(0) - total_wait(T) - switch_cost * num_switches

    i.e. maximizing return is *exactly* equivalent to minimizing net
    waiting time accumulated plus a small switching tax — there is no
    separate, independently-tuned term that could fight the primary
    objective. Because wait time is part of the state, a starved lane's
    growing wait keeps depressing future reward until it is served, so
    fairness/starvation-avoidance emerges from this single term instead
    of needing a hand-tuned starvation bonus.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ENV_CONFIG


class _Space:
    """Minimal gym-space stand-in (shape for Box, n for Discrete)."""
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class TrafficEnv:

    DIRECTIONS = ENV_CONFIG.directions
    PHASE_GROUPS = {0: [0, 2], 1: [1, 3]}   # 0=right+left, 1=down+up

    def __init__(
        self,
        max_queue: int = ENV_CONFIG.max_queue,
        arrival_rate: float = ENV_CONFIG.arrival_rate,
        release_capacity: int = ENV_CONFIG.release_capacity,
        min_green_steps: int = ENV_CONFIG.min_green_steps,
        max_wait_norm: float = ENV_CONFIG.max_wait_norm,
        switch_cost: float = ENV_CONFIG.switch_cost,
        max_steps: int = ENV_CONFIG.max_steps_per_episode,
        seed: Optional[int] = None,
    ):
        self.max_queue = max_queue
        self.arrival_rate = arrival_rate
        self.release_capacity = release_capacity
        self.min_green_steps = min_green_steps
        self.max_wait_norm = max_wait_norm
        self.switch_cost = switch_cost
        self.max_steps = max_steps

        self.observation_space = _Space(shape=(10,))
        self.action_space = _Space(n=2)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ---------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.queues: List[int] = [0, 0, 0, 0]
        self.waits: List[int] = [0, 0, 0, 0]
        self.phase: int = 0
        self.phase_elapsed: int = 0
        self.step_num: int = 0

        # Cumulative episode metrics (spec section 10)
        self.total_waiting_time: float = 0.0
        self.vehicles_served: int = 0
        self.num_phase_changes: int = 0

        self._state = self._build_state()
        return self._state, {}

    # ----------------------------------------------------------------- step
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert action in (0, 1), f"invalid action {action}"

        wait_before = float(sum(self.waits))

        # 1. Resolve the (possibly masked) phase transition.
        switched = False
        if action == 1 and self.phase_elapsed >= self.min_green_steps:
            self.phase = 1 - self.phase
            self.phase_elapsed = 0
            switched = True
        # else: keep current phase (either action==0, or an illegal early
        # switch request that the environment masks to a no-op)

        green_idx = self.PHASE_GROUPS[self.phase]

        # 2. Release vehicles from green lanes.
        released = 0
        for i in green_idx:
            take = min(self.queues[i], self.release_capacity)
            self.queues[i] -= take
            released += take
        self.vehicles_served += released

        # 3. New arrivals (Bernoulli per direction, capped at max_queue).
        for i in range(4):
            if self._rng.random() < self.arrival_rate:
                self.queues[i] = min(self.queues[i] + 1, self.max_queue)

        # 4. Update wait timers: served lanes reset, others accrue.
        for i in range(4):
            self.waits[i] = 0 if i in green_idx else self.waits[i] + 1

        wait_after = float(sum(self.waits))
        self.total_waiting_time += wait_after

        # 5. Reward: reduction in total waiting time, minus a switching tax.
        reward = (wait_before - wait_after) - (self.switch_cost if switched else 0.0)

        self.phase_elapsed += 1
        self.step_num += 1
        if switched:
            self.num_phase_changes += 1

        done = self.step_num >= self.max_steps
        self._state = self._build_state()

        info = {
            "queues": list(self.queues),
            "waits": list(self.waits),
            "released": released,
            "switched": switched,
            "green_dirs": [self.DIRECTIONS[i] for i in green_idx],
            "total_waiting_time": self.total_waiting_time,
            "vehicles_served": self.vehicles_served,
            "num_phase_changes": self.num_phase_changes,
            "avg_queue": sum(self.queues) / 4.0,
            "max_queue_len": max(self.queues),
        }
        return self._state, reward, done, False, info

    # ------------------------------------------------------------- helpers
    def _build_state(self) -> np.ndarray:
        q_norm = [q / self.max_queue for q in self.queues]
        w_norm = [min(w / self.max_wait_norm, 1.0) for w in self.waits]
        phase_norm = float(self.phase)
        elapsed_norm = min(self.phase_elapsed / max(self.min_green_steps, 1), 1.0)
        return np.array(q_norm + w_norm + [phase_norm, elapsed_norm], dtype=np.float32)

    def vehicle_counts(self) -> Dict[str, int]:
        return {d: self.queues[i] for i, d in enumerate(self.DIRECTIONS)}

    def is_switch_legal(self) -> bool:
        return self.phase_elapsed >= self.min_green_steps
