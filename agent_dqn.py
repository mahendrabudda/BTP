"""
agent_dqn.py
============
The Deep Q-Network and the agent that trains it.

Implements standard DQN (spec section 5):
    1. Online Q-network                     -> self.policy_net
    2. Target Q-network                     -> self.target_net
    3. Replay buffer                        -> injected, see replay_buffer.py
    4. Epsilon-greedy action selection       -> select_action()
    5. Mini-batch sampling                   -> buffer.sample()
    6. Bellman target: y = r + gamma * max_a' Q_target(s', a')  (0 if done)
    7. Huber (Smooth L1) loss
    8-9. Backprop + Adam optimizer
    10. Hard target-network updates every `target_update_frequency` steps
    12. Terminal states are NOT bootstrapped (multiplied by (1 - done))

Optionally implements Double DQN (spec section 18), which decouples
action *selection* (online network) from action *evaluation* (target
network) to reduce Q-value overestimation:
    a*      = argmax_a' Q_online(s', a')
    target  = r + gamma * Q_target(s', a*)
This is OFF by default (config.DQN_CONFIG.use_double_dqn) — standard DQN
should be validated first.
"""

import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """MLP that maps a state vector to one Q-value per discrete action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        hidden_sizes: List[int] = (128, 128),
        lr: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer: Optional[ReplayBuffer] = None,
        min_replay_size: int = 1000,
        target_update_frequency: int = 500,
        use_double_dqn: bool = False,
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = buffer
        self.min_replay_size = min_replay_size
        self.target_update_frequency = target_update_frequency
        self.use_double_dqn = use_double_dqn

        self.policy_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is never trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, per spec section 5

        self.update_count = 0

        self._sanity_check_init(state_dim, action_dim)

    # ------------------------------------------------------------ sanity
    def _sanity_check_init(self, state_dim: int, action_dim: int) -> None:
        """Spec section 16: verify shapes and target/online parity before training."""
        dummy = torch.zeros((1, state_dim), device=self.device)
        out = self.policy_net(dummy)
        assert out.shape == (1, action_dim), (
            f"QNetwork output dim {out.shape[-1]} != action_dim {action_dim}"
        )
        for p_online, p_target in zip(self.policy_net.parameters(), self.target_net.parameters()):
            assert torch.equal(p_online, p_target), "target_net does not match policy_net at init"

    # ------------------------------------------------------ action selection
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy over Q(s, ·). epsilon=0.0 -> fully greedy (use for eval)."""
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def get_q_values(self, state: np.ndarray) -> List[float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.squeeze(0).tolist()

    # ------------------------------------------------------------- training
    def train_step(self) -> Optional[float]:
        """One gradient step. Returns the loss, or None if not enough data yet."""
        if self.buffer is None or len(self.buffer) < max(self.batch_size, self.min_replay_size):
            return None

        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device)

        # Current Q(s, a) for the actions actually taken.
        q_values = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)

        # Bellman target — computed with no grad so the target network
        # (and the online network, via next-state input) is never trained
        # through this path.
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.policy_net(next_state).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_state).max(dim=1)[0]
            target_q = reward + (1.0 - done) * self.gamma * next_q

        assert not torch.isnan(q_values).any(), "NaN in predicted Q-values"
        assert not torch.isnan(target_q).any(), "NaN in Bellman targets"

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # ----------------------------------------------------------- persistence
    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path: str) -> None:
        if not os.path.exists(path):
            print(f"[DQNAgent] No checkpoint at '{path}' — using randomly initialized weights.")
            return
        state_dict = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        print(f"[DQNAgent] Loaded weights <- {path}")
