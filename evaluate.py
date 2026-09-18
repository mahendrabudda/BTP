"""
evaluate.py
===========
Compares the trained DQN controller against Fixed-Time, Random, and
Actuated baselines on identical traffic scenarios (spec section 12).

During evaluation:
  - epsilon = 0 for the DQN (no exploration)
  - the network is never updated (no train_step calls, no grad)
  - every controller sees the SAME sequence of per-episode seeds, so
    arrivals are identical across controllers within a given episode
    index — differences in outcome are due to the controller, not luck.

Usage
-----
    python evaluate.py --model models/dqn_checkpoint_final.pt
"""

import argparse

import numpy as np
import torch

from agent_dqn import QNetwork
from baselines import ActuatedController, FixedTimeController, RandomController
from config import DQN_CONFIG, ENV_CONFIG, EVAL_CONFIG
from traffic_env import TrafficEnv
from utils import plot_controller_comparison, set_seed

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw):
        return range(n)


def run_episodes(controller_name: str, act_fn, episodes: int, max_steps: int, base_seed: int, reset_fn=None):
    """
    `act_fn(env, state)` -> action.  Runs `episodes` episodes, each with
    seed = base_seed + episode_index, and returns per-episode metrics.
    """
    env = TrafficEnv(
        max_queue=ENV_CONFIG.max_queue,
        arrival_rate=ENV_CONFIG.arrival_rate,
        release_capacity=ENV_CONFIG.release_capacity,
        min_green_steps=ENV_CONFIG.min_green_steps,
        max_wait_norm=ENV_CONFIG.max_wait_norm,
        switch_cost=ENV_CONFIG.switch_cost,
        max_steps=max_steps,
    )

    rewards, avg_waits, avg_queues, throughputs, phase_changes = [], [], [], [], []

    for ep in trange(episodes, desc=controller_name):
        state, _ = env.reset(seed=base_seed + ep)
        if reset_fn is not None:
            reset_fn()
        ep_reward = 0.0
        last_info = {}

        for _ in range(max_steps):
            action = act_fn(env, state)
            state, reward, done, _, info = env.step(action)
            ep_reward += reward
            last_info = info
            if done:
                break

        n_steps = env.step_num
        rewards.append(ep_reward)
        avg_waits.append(last_info.get("total_waiting_time", 0.0) / max(n_steps, 1))
        avg_queues.append(last_info.get("avg_queue", 0.0))
        throughputs.append(last_info.get("vehicles_served", 0))
        phase_changes.append(last_info.get("num_phase_changes", 0))

    return {
        "reward_mean": np.mean(rewards), "reward_std": np.std(rewards),
        "avg_wait": np.mean(avg_waits), "avg_wait_std": np.std(avg_waits),
        "avg_queue": np.mean(avg_queues), "avg_queue_std": np.std(avg_queues),
        "throughput": np.mean(throughputs), "throughput_std": np.std(throughputs),
        "phase_changes": np.mean(phase_changes),
    }


def evaluate(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    state_dim = TrafficEnv().observation_space.shape[0]
    action_dim = TrafficEnv().action_space.n

    net = QNetwork(state_dim, action_dim, DQN_CONFIG.hidden_sizes)
    net.load_state_dict(torch.load(args.model, map_location="cpu"))
    net.eval()

    def dqn_act(env, state):
        with torch.no_grad():
            q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())  # epsilon = 0: fully greedy

    fixed = FixedTimeController(phase_duration=args.fixed_phase_duration)
    random_ctrl = RandomController(seed=args.seed)
    actuated = ActuatedController()

    controllers = {
        "Fixed-Time": (lambda env, s: fixed.select_action(env), fixed.reset),
        "Random": (lambda env, s: random_ctrl.select_action(env), random_ctrl.reset),
        "Actuated": (lambda env, s: actuated.select_action(env), actuated.reset),
        "DQN": (dqn_act, None),
    }

    results = {}
    for name, (act_fn, reset_fn) in controllers.items():
        results[name] = run_episodes(name, act_fn, args.episodes, args.max_steps, args.seed, reset_fn)

    header = f"{'Controller':<12} {'Avg Wait':>14} {'Avg Queue':>14} {'Throughput':>14} {'Phase Chg':>10}"
    print("\n" + header)
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<12} "
            f"{r['avg_wait']:>8.2f} ± {r['avg_wait_std']:<4.2f} "
            f"{r['avg_queue']:>8.2f} ± {r['avg_queue_std']:<4.2f} "
            f"{r['throughput']:>8.1f} ± {r['throughput_std']:<4.1f} "
            f"{r['phase_changes']:>10.1f}"
        )
    print()

    plot_data = {name: {"avg_wait": r["avg_wait"], "avg_queue": r["avg_queue"], "throughput": r["throughput"]}
                 for name, r in results.items()}
    plot_controller_comparison(plot_data, filename="controller_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG.episodes)
    parser.add_argument("--max_steps", type=int, default=EVAL_CONFIG.max_steps)
    parser.add_argument("--seed", type=int, default=EVAL_CONFIG.seed)
    parser.add_argument("--fixed_phase_duration", type=int, default=10)
    evaluate(parser.parse_args())
