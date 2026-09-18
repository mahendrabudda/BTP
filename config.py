"""
config.py
=========
Centralized configuration for the DQN traffic-signal controller.

Every hyperparameter that affects the environment, the network, or
training lives here so nothing is hardcoded deep inside other modules
(spec section 14). Import this module and read/override fields on the
dataclass instances rather than editing constants scattered around the
codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    """Traffic environment dynamics."""
    directions: List[str] = field(default_factory=lambda: ["right", "down", "left", "up"])

    max_queue: int = 20          # queue length that saturates the normalized state (vehicles)
    arrival_rate: float = 0.35   # P(one new vehicle arrives) per direction per step
    release_capacity: int = 2    # max vehicles released per green direction per step

    min_green_steps: int = 5     # a phase must run this many steps before a switch is legal
    max_wait_norm: float = 60.0  # wait time (steps) that saturates the normalized state
    switch_cost: float = 2.0     # fixed reward penalty applied when a phase switch occurs

    max_steps_per_episode: int = 200


@dataclass
class DQNConfig:
    """Network architecture and algorithm settings."""
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    use_double_dqn: bool = False   # standard DQN first; flip on only after it's validated

    learning_rate: float = 1e-3
    gamma: float = 0.95   # episodes are short (200 steps); a long horizon makes the
                           # Bellman target's constant baseline dominate the actual
                           # action-relevant signal, slowing learning (validated empirically)
    reward_scale: float = 0.1   # multiplies reward ONLY when pushed into the replay
                                 # buffer (numerical conditioning for the optimizer).
                                 # traffic_env.py's returned reward is untouched — this
                                 # does not change the objective, only its scale.
    batch_size: int = 64
    replay_buffer_size: int = 20_000
    min_replay_size: int = 1_000   # do not start optimizing before this many transitions exist

    target_update_frequency: int = 500   # hard update, in optimizer steps

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5_000     # exponential decay time-constant, in env steps


@dataclass
class TrainConfig:
    episodes: int = 600
    save_every_episodes: int = 50
    save_dir: str = "models"
    checkpoint_name: str = "dqn_checkpoint_final.pt"
    seed: int = 0
    device: str = "auto"   # "auto" | "cpu" | "cuda"


@dataclass
class EvalConfig:
    episodes: int = 50
    seed: int = 123          # different from training seed; scenarios are still identical
                              # *across controllers* within an eval run (see evaluate.py)
    max_steps: int = 200


ENV_CONFIG = EnvConfig()
DQN_CONFIG = DQNConfig()
TRAIN_CONFIG = TrainConfig()
EVAL_CONFIG = EvalConfig()
