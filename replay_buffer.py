"""
replay_buffer.py
=================
Standard uniform-sampling experience replay buffer, independent of any
neural-network code (spec section 6). Stores (s, a, r, s', done) tuples
and returns them as stacked numpy arrays for the agent to convert to
tensors.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self) -> int:
        return len(self.buffer)
