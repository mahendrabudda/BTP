"""
train.py
========
Headless DQN training loop.

    State -> DQN -> Action -> TrafficEnv -> (Reward, Next State) -> Replay Buffer -> DQN update

Usage
-----
    python train.py
    python train.py --episodes 1000 --device cuda
"""

import argparse
import math
import os

from agent_dqn import DQNAgent
from baselines import FixedTimeController
from config import DQN_CONFIG, ENV_CONFIG, TRAIN_CONFIG
from replay_buffer import ReplayBuffer
from traffic_env import TrafficEnv
from utils import get_device, plot_training_metrics, set_seed

try:
    from tqdm import trange
except ImportError:
    def trange(n, **kw):
        return range(n)


def sanity_check_environment(env: TrafficEnv, agent: DQNAgent) -> None:
    """Spec section 16: pre-flight checks before training starts."""
    state, _ = env.reset(seed=0)
    assert state.shape == env.observation_space.shape, "state shape mismatch"
    assert env.action_space.n == 2, "unexpected action space size"

    next_state, reward, done, _, info = env.step(0)
    assert next_state.shape == state.shape, "next_state shape mismatch"
    assert isinstance(reward, float), "reward must be a python float"
    assert not math.isnan(reward), "reward is NaN"
    assert isinstance(done, bool), "done must be a bool"

    # A fixed-time controller run for a handful of steps must never crash
    # and must always propose a legal action.
    fixed = FixedTimeController(phase_duration=5)
    fixed.reset()
    env.reset(seed=0)
    for _ in range(20):
        a = fixed.select_action(env)
        assert a in (0, 1), "baseline controller produced an invalid action"
        env.step(a)

    print("[sanity] All pre-training checks passed.")


def train(args: argparse.Namespace) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = get_device(args.device)

    env = TrafficEnv(
        max_queue=ENV_CONFIG.max_queue,
        arrival_rate=ENV_CONFIG.arrival_rate,
        release_capacity=ENV_CONFIG.release_capacity,
        min_green_steps=ENV_CONFIG.min_green_steps,
        max_wait_norm=ENV_CONFIG.max_wait_norm,
        switch_cost=ENV_CONFIG.switch_cost,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    buffer = ReplayBuffer(capacity=DQN_CONFIG.replay_buffer_size)
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        device=device,
        hidden_sizes=DQN_CONFIG.hidden_sizes,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer=buffer,
        min_replay_size=DQN_CONFIG.min_replay_size,
        target_update_frequency=args.target_update,
        use_double_dqn=DQN_CONFIG.use_double_dqn,
    )

    sanity_check_environment(env, agent)

    eps_start, eps_end, eps_decay = args.eps_start, args.eps_end, args.eps_decay

    history = {
        "reward": [], "avg_wait": [], "avg_queue": [], "max_queue": [],
        "throughput": [], "phase_changes": [], "epsilon": [], "loss": [],
    }
    global_step = 0

    print(f"\n{'=' * 55}\n  DQN Traffic Controller — Training")
    print(f"  Episodes : {args.episodes}  |  Steps/ep : {args.max_steps}")
    print(f"  Device   : {device}\n{'=' * 55}\n")

    for ep in trange(args.episodes, desc="Episodes"):
        state, _ = env.reset(seed=args.seed + ep)  # vary traffic pattern per episode
        ep_reward = 0.0
        ep_losses = []
        last_info = {}

        for _ in range(args.max_steps):
            epsilon = eps_end + (eps_start - eps_end) * math.exp(-1.0 * global_step / eps_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, info = env.step(action)

            # reward_scale only affects what the optimizer sees (numerical
            # conditioning); ep_reward below tracks the true, unscaled reward.
            buffer.push(state, action, reward * DQN_CONFIG.reward_scale, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
                history["loss"].append(loss)

            state = next_state
            ep_reward += reward
            last_info = info
            global_step += 1
            if done:
                break

        n_steps = env.step_num
        history["reward"].append(ep_reward)
        history["avg_wait"].append(last_info.get("total_waiting_time", 0.0) / max(n_steps, 1))
        history["avg_queue"].append(last_info.get("avg_queue", 0.0))
        history["max_queue"].append(last_info.get("max_queue_len", 0))
        history["throughput"].append(last_info.get("vehicles_served", 0))
        history["phase_changes"].append(last_info.get("num_phase_changes", 0))
        history["epsilon"].append(epsilon)

        if (ep + 1) % args.save_every == 0:
            ckpt = os.path.join(args.save_dir, f"dqn_checkpoint_ep{ep + 1}.pt")
            agent.save_model(ckpt)
            recent_r = sum(history["reward"][-20:]) / min(len(history["reward"]), 20)
            print(f"\n  [save] {ckpt}  (avg reward, last 20 eps: {recent_r:.1f})")

    final_path = os.path.join(args.save_dir, TRAIN_CONFIG.checkpoint_name)
    agent.save_model(final_path)
    plot_training_metrics(history, filename="training_results.png")
    print(f"\n[train] Done. Model saved -> {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=TRAIN_CONFIG.episodes)
    parser.add_argument("--max_steps", type=int, default=ENV_CONFIG.max_steps_per_episode)
    parser.add_argument("--save_dir", type=str, default=TRAIN_CONFIG.save_dir)
    parser.add_argument("--batch_size", type=int, default=DQN_CONFIG.batch_size)
    parser.add_argument("--lr", type=float, default=DQN_CONFIG.learning_rate)
    parser.add_argument("--gamma", type=float, default=DQN_CONFIG.gamma)
    parser.add_argument("--target_update", type=int, default=DQN_CONFIG.target_update_frequency)
    parser.add_argument("--eps_start", type=float, default=DQN_CONFIG.epsilon_start)
    parser.add_argument("--eps_end", type=float, default=DQN_CONFIG.epsilon_end)
    parser.add_argument("--eps_decay", type=int, default=DQN_CONFIG.epsilon_decay_steps)
    parser.add_argument("--save_every", type=int, default=TRAIN_CONFIG.save_every_episodes)
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG.seed)
    parser.add_argument("--device", type=str, default=TRAIN_CONFIG.device)
    train(parser.parse_args())
