"""
visualize.py
============
Lightweight Pygame animation of a trained (frozen) DQN policy running
against the headless TrafficEnv — useful for a quick qualitative look
without the full vehicle-sprite simulation.py.

Usage
-----
    python visualize.py --model models/dqn_checkpoint_final.pt
"""

import argparse

import pygame
import torch

from agent_dqn import QNetwork
from config import DQN_CONFIG
from traffic_env import TrafficEnv


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    screen.blit(font.render(text, True, color), (x, y))


def draw_static_intersection(screen, env, action, step, reward, episode, font):
    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (60, 60, 60), (250, 250, 100, 100))

    # Phase 0 = horizontal (right+left) green, Phase 1 = vertical (down+up) green.
    horiz_color = (0, 255, 0) if env.phase == 0 else (255, 0, 0)
    vert_color  = (0, 255, 0) if env.phase == 1 else (255, 0, 0)

    pygame.draw.circle(screen, horiz_color, (220, 300), 10)  # left
    pygame.draw.circle(screen, horiz_color, (380, 300), 10)  # right
    pygame.draw.circle(screen, vert_color,  (300, 220), 10)  # up
    pygame.draw.circle(screen, vert_color,  (300, 380), 10)  # down

    q_right, q_down, q_left, q_up = env.queues
    scale = 10
    pygame.draw.rect(screen, (0, 150, 255), (295, 200 - q_up   * scale, 10, q_up   * scale))
    pygame.draw.rect(screen, (0, 150, 255), (295, 350,                  10, q_down * scale))
    pygame.draw.rect(screen, (0, 150, 255), (350, 295, q_right * scale, 10))
    pygame.draw.rect(screen, (0, 150, 255), (200 - q_left * scale, 295, q_left * scale, 10))

    draw_text(screen, f"Episode : {episode}",   10,  10, font)
    draw_text(screen, f"Step    : {step}",       10,  40, font)
    draw_text(screen, f"Reward  : {reward:.2f}", 10,  70, font)
    action_lbl = "Keep phase" if action == 0 else "Switch phase"
    draw_text(screen, f"Action  : {action_lbl}", 10, 100, font, (0, 255, 0))
    draw_text(screen, f"Queues  R:{q_right} D:{q_down} L:{q_left} U:{q_up}", 10, 130, font)
    draw_text(screen, f"Waits   R:{env.waits[0]} D:{env.waits[1]} L:{env.waits[2]} U:{env.waits[3]}", 10, 160, font)

    pygame.display.flip()


def run_visualization(model_path, episodes=3, max_steps=300, fps=10):
    env = TrafficEnv(max_steps=max_steps)
    net = QNetwork(env.observation_space.shape[0], env.action_space.n, DQN_CONFIG.hidden_sizes)
    net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Traffic DQN Visualization")
    font  = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    running = True
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(1, max_steps + 1):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break

            with torch.no_grad():
                q = net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            action = int(q.argmax(dim=1).item())   # frozen policy: epsilon = 0

            next_state, reward, done, _, info = env.step(action)
            draw_static_intersection(screen, env, action, step, reward, ep, font)
            state         = next_state
            total_reward += reward
            clock.tick(fps)
            if done:
                break

        print(f"Episode {ep}: Total Reward = {total_reward:.2f}")
        if not running:
            break

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Traffic DQN Agent")
    parser.add_argument("--model",     type=str, required=True)
    parser.add_argument("--episodes",  type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--fps",       type=int, default=10)
    args = parser.parse_args()
    run_visualization(args.model, args.episodes, args.max_steps, args.fps)
