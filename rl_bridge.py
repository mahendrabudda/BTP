"""
rl_bridge.py
============
Adapter between the Pygame simulation (simulation.py) and the DQN agent.

Design change from the previous version
----------------------------------------
The old bridge computed its own reward function and its own
starvation/monopoly rules, independent from (and inconsistent with)
traffic_env.py — and those rules could override argmax(Q) outright,
meaning the "DQN controller" was really a rule-based controller most of
the time. That is fixed here:

  * State construction mirrors traffic_env.py's TrafficEnv._build_state
    exactly (same 10-dim layout, same normalization constants, pulled
    from config.py) so a checkpoint trained by train.py behaves
    identically when deployed here.
  * There is no rule that overrides the policy's chosen action. Legal-
    action masking (can't switch before MIN_GREEN) is enforced the same
    way TrafficEnv enforces it — as an environment constraint, not a
    policy override — so the DQN is the only thing deciding *which*
    legal action to take.
  * Online learning (fine-tuning the checkpoint live during the Pygame
    demo) is OFF by default. Spec section 17 explicitly warns against
    mixing training and deployment; enabling it here is opt-in and
    clearly logged as a non-standard mode, not the default behavior.
  * A lock guards the fields read by the render thread (`metrics`)
    against the signal-control thread that writes them.
"""

import math
import threading
from typing import Dict, List, Optional

import numpy as np

from agent_dqn import DQNAgent
from config import DQN_CONFIG, ENV_CONFIG
from replay_buffer import ReplayBuffer

DIRECTIONS = ENV_CONFIG.directions
PHASE_GROUPS = {0: [0, 2], 1: [1, 3]}   # 0 = right+left green, 1 = down+up green
STATE_DIM = 10
ACTION_DIM = 2


class RLBridge:

    def __init__(
        self,
        model_path: str = "models/dqn_checkpoint_final.pt",
        online_learning: bool = False,
        save_path: str = "models/dqn_checkpoint_final.pt",
        save_every: int = 500,
    ):
        self.online = online_learning
        self.save_every = save_every
        self.save_path = save_path

        self.buffer = ReplayBuffer(capacity=DQN_CONFIG.replay_buffer_size) if online_learning else None
        self.agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=DQN_CONFIG.hidden_sizes,
            lr=DQN_CONFIG.learning_rate,
            gamma=DQN_CONFIG.gamma,
            batch_size=DQN_CONFIG.batch_size,
            buffer=self.buffer,
            min_replay_size=DQN_CONFIG.min_replay_size,
            target_update_frequency=DQN_CONFIG.target_update_frequency,
            use_double_dqn=DQN_CONFIG.use_double_dqn,
        )
        self.agent.load_model(model_path)
        if not online_learning:
            self.agent.policy_net.eval()

        if online_learning:
            print(
                "[RLBridge] online_learning=True — the network will keep training "
                "during the live demo. This is a non-standard/experimental mode; "
                "the checkpoints used for evaluate.py comparisons should come from "
                "train.py, not from this live loop."
            )

        # Environment-consistent bookkeeping (mirrors TrafficEnv's internal state).
        self.phase: int = 0
        self.phase_elapsed_steps: int = 0
        self.waits: List[float] = [0.0, 0.0, 0.0, 0.0]

        self._lock = threading.Lock()
        self._prev_state: Optional[np.ndarray] = None
        self._prev_action: Optional[int] = None
        self._last_qvals: List[float] = [0.0, 0.0]
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._last_reward: float = 0.0
        self._global_step: int = 0

    # ── public API used by simulation.py ────────────────────────────────
    def choose(self, vehicles, last_cycle_seconds: float = 1.0) -> int:
        """
        Called once per signal decision point. Returns action in {0, 1}
        using the SAME "keep phase / switch phase" semantics as
        TrafficEnv, with switching masked illegal before MIN_GREEN.
        `last_cycle_seconds` advances the internal step/wait bookkeeping
        (treated as whole decision steps, matching the training env).
        """
        counts = self._count_waiting(vehicles)
        state = self._build_state(counts)

        legal_switch = self.phase_elapsed_steps >= ENV_CONFIG.min_green_steps
        action = self.agent.select_action(state, epsilon=0.0)   # deployment: greedy, epsilon=0
        if action == 1 and not legal_switch:
            action = 0   # environment-level masking, not a policy override

        with self._lock:
            self._last_qvals = self.agent.get_q_values(state)
            self._prev_state = state
            self._prev_action = action

        if action == 1:
            self.phase = 1 - self.phase
            self.phase_elapsed_steps = 0
        else:
            self.phase_elapsed_steps += 1

        green_idx = PHASE_GROUPS[self.phase]
        for i, d in enumerate(DIRECTIONS):
            self.waits[i] = 0.0 if i in green_idx else self.waits[i] + last_cycle_seconds

        return action

    def feedback(self, vehicles) -> None:
        """
        Called periodically during a green phase. Only does work (reward
        computation, replay push, optimizer step) when online_learning
        is enabled — otherwise this is a no-op, keeping deployment and
        training cleanly separated.
        """
        if not self.online or self._prev_state is None:
            return

        counts = self._count_waiting(vehicles)
        state = self._build_state(counts)
        reward = self._prev_wait_sum - sum(self.waits)   # same telescoping identity as TrafficEnv

        with self._lock:
            self.buffer.push(self._prev_state, self._prev_action, reward, state, False)
            loss = self.agent.train_step()
            self._prev_state = state
            self._last_reward = reward
            self._total_reward += reward
            self._step_count += 1
            self._global_step += 1

        if self._global_step % self.save_every == 0:
            self.agent.save_model(self.save_path)

    # ── internal helpers ─────────────────────────────────────────────
    @property
    def _prev_wait_sum(self) -> float:
        return sum(self.waits)

    def _build_state(self, counts: Dict[str, int]) -> np.ndarray:
        q_norm = [counts[d] / ENV_CONFIG.max_queue for d in DIRECTIONS]
        w_norm = [min(w / ENV_CONFIG.max_wait_norm, 1.0) for w in self.waits]
        phase_norm = float(self.phase)
        elapsed_norm = min(self.phase_elapsed_steps / max(ENV_CONFIG.min_green_steps, 1), 1.0)
        return np.array(q_norm + w_norm + [phase_norm, elapsed_norm], dtype=np.float32)

    def _count_waiting(self, vehicles) -> Dict[str, int]:
        result = {}
        for d in DIRECTIONS:
            total = 0
            for lane in range(3):
                for v in vehicles[d][lane]:
                    if v.crossed == 0:
                        total += 1
            result[d] = total
        return result

    # ── HUD telemetry (read by the render thread) ───────────────────
    @property
    def metrics(self) -> dict:
        with self._lock:
            q2 = self._last_qvals
            return {
                "state": tuple(self._prev_state.tolist()) if self._prev_state is not None else tuple([0.0] * STATE_DIM),
                "action": self._prev_action if self._prev_action is not None else 0,
                "reward": self._last_reward,
                "total_reward": self._total_reward,
                "steps": self._step_count,
                "q_values": [q2[0], q2[1], q2[0], q2[1]],
                "wait_steps": list(self.waits),
                "online_learning": self.online,
            }
