"""
Replay collected dataset in the IsaacGym simulator.
Loads actions from .npy files and steps the environment to visualize successful episodes.

Usage:
    python replay_dataset.py --task=ShadowHandSwingCup --data_dir=./logs/ShadowHandSwingCup/ppo/ppo_seed-1/model_800 --num_envs=1 --episode=0
    python replay_dataset.py --task=ShadowHandSwingCup --data_dir=./logs/ShadowHandSwingCup/ppo/ppo_seed-1/model_800 --num_envs=1 --episode=all
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bidexhands'))

# isaacgym must be imported before torch
from isaacgym import gymapi
from isaacgym import gymtorch

import numpy as np
import torch

from bidexhands.utils.config import set_np_formatting, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex


def split_episodes(actions, dones):
    """Split concatenated transitions into per-episode action sequences."""
    episodes = []
    ep_start = 0
    done_indices = np.where(dones.flatten() > 0)[0]
    for done_idx in done_indices:
        ep_end = done_idx + 1
        episodes.append(actions[ep_start:ep_end])
        ep_start = ep_end
    # leftover (incomplete episode)
    if ep_start < len(actions):
        episodes.append(actions[ep_start:])
    return episodes


def main():
    # Parse custom args first, then pass the rest to isaacgym
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', type=str, required=True, help='Path to directory with .npy files')
    parser.add_argument('--episode', type=str, default='0', help='Episode index to replay, or "all" for all episodes')
    parser.add_argument('--playback_speed', type=float, default=1.0, help='Playback speed multiplier (< 1 = slower)')
    known, remaining = parser.parse_known_args()

    # Override sys.argv for isaacgym arg parsing
    sys.argv = [sys.argv[0]] + remaining

    # Load dataset
    data_dir = known.data_dir
    print(f"Loading dataset from: {data_dir}")
    actions_all = np.load(os.path.join(data_dir, 'actions.npy'))
    dones_all = np.load(os.path.join(data_dir, 'dones.npy'))
    rewards_all = np.load(os.path.join(data_dir, 'rewards.npy'))

    episodes = split_episodes(actions_all, dones_all)
    ep_rewards = split_episodes(rewards_all, dones_all)
    print(f"Total transitions: {len(actions_all)}")
    print(f"Total episodes: {len(episodes)}")
    for i, ep in enumerate(episodes[:10]):
        r = ep_rewards[i].sum()
        print(f"  Episode {i}: {len(ep)} steps, total reward: {r:.1f}")
    if len(episodes) > 10:
        print(f"  ... and {len(episodes) - 10} more episodes")

    # Determine which episodes to play
    if known.episode == 'all':
        ep_indices = list(range(len(episodes)))
    else:
        ep_indices = [int(known.episode)]

    # Setup isaacgym environment (headless=False for visualization)
    set_np_formatting()
    args = get_args()
    args.num_envs = 1
    args.headless = False
    args.test = True
    cfg, cfg_train, logdir = load_cfg(args)
    cfg["env"]["numEnvs"] = 1
    sim_params = parse_sim_params(args, cfg, cfg_train)
    agent_index = get_AgentIndex(cfg)

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    sleep_time = (1.0 / 60.0) / known.playback_speed

    print("\n=== Replay Controls ===")
    print("Close the viewer window to stop.")
    print(f"Playback speed: {known.playback_speed}x")
    print("========================\n")

    for ep_idx in ep_indices:
        if ep_idx >= len(episodes):
            print(f"Episode {ep_idx} does not exist (max: {len(episodes)-1})")
            break

        ep_actions = episodes[ep_idx]
        ep_reward = ep_rewards[ep_idx].sum()
        print(f"\n>>> Playing Episode {ep_idx}/{len(episodes)-1}: {len(ep_actions)} steps, total reward: {ep_reward:.1f}")

        # Reset environment
        env.reset()

        for step_idx, action in enumerate(ep_actions):
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(env.rl_device)
            obs, rew, done, info = env.step(action_tensor)

            time.sleep(sleep_time)

            if step_idx % 50 == 0:
                print(f"  Step {step_idx}/{len(ep_actions)}, reward: {rew.item():.3f}")

        print(f"<<< Episode {ep_idx} done. Steps: {len(ep_actions)}, Total reward: {ep_reward:.1f}")

        # Pause between episodes
        if ep_idx != ep_indices[-1]:
            print("  (Next episode in 2 seconds...)")
            time.sleep(2.0)

    print("\nReplay finished.")


if __name__ == '__main__':
    main()
