# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Evaluation script: loads a trained model and reports reward/success metrics.

import numpy as np
from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_sarl import process_sarl
from bidexhands.utils.process_marl import process_MultiAgentRL, get_AgentIndex
import torch

SARL_ALGOS = ["ppo", "ddpg", "sac", "td3", "trpo"]


def evaluate():
    print(f"=== Evaluation: {args.task} / {args.algo} ===")
    agent_index = get_AgentIndex(cfg)

    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    model = process_sarl(args, env, cfg_train, logdir)

    assert args.model_dir != "", "Must provide --model_dir for evaluation"
    print(f"Loading model from {args.model_dir}")
    model.test(args.model_dir)

    num_envs = env.num_envs
    num_eval_episodes = 5  # each env runs this many episodes
    max_steps = cfg["env"].get("episodeLength", 300) * num_eval_episodes

    current_obs = env.reset()

    # per-env trackers
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=model.device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=model.device)

    episode_rewards = []
    episode_lengths = []
    episode_successes = []

    print(f"Running evaluation: {num_envs} envs x {num_eval_episodes} episodes (max {max_steps} steps)...")

    for step in range(max_steps):
        with torch.no_grad():
            actions = model.actor_critic.act_inference(current_obs)
            next_obs, rews, dones, infos = env.step(actions)
            current_obs.copy_(next_obs)

        cur_reward_sum += rews.squeeze()
        cur_episode_length += 1

        # check done envs
        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            for idx in done_ids:
                i = idx.item()
                ep_rew = cur_reward_sum[i].item()
                ep_len = cur_episode_length[i].item()
                episode_rewards.append(ep_rew)
                episode_lengths.append(ep_len)

                # get success info
                success_flag = None
                if 'successes' in infos:
                    s = infos['successes'][i].item()
                    episode_successes.append(s)
                    success_flag = s > 0

                ep_num = len(episode_rewards)
                success_str = f", Success: {'O' if success_flag else 'X'}" if success_flag is not None else ""
                print(f"  [Episode {ep_num}] Env {i} | Reward: {ep_rew:.2f} | Length: {ep_len:.0f}{success_str}")

            cur_reward_sum[done_ids] = 0
            cur_episode_length[done_ids] = 0

        total_episodes = len(episode_rewards)
        if total_episodes >= num_envs * num_eval_episodes:
            break

        if (step + 1) % 1000 == 0:
            sr_str = ""
            if episode_successes:
                sr = np.mean([1.0 if s > 0 else 0.0 for s in episode_successes]) * 100
                sr_str = f", Success rate: {sr:.1f}%"
            print(f"  Step {step+1}/{max_steps}, Episodes completed: {total_episodes}{sr_str}")

    # Results
    total_episodes = len(episode_rewards)
    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {args.task}")
    print(f"{'='*60}")
    print(f"  Total episodes:          {total_episodes}")
    print(f"  Mean reward:             {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean episode length:     {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")

    if episode_successes:
        success_rate = np.mean([1.0 if s > 0 else 0.0 for s in episode_successes])
        print(f"  Success rate:            {success_rate*100:.1f}%")
        print(f"  Mean success value:      {np.mean(episode_successes):.4f}")

        # 성공한 에피소드들만의 리워드 통계
        success_rewards = [r for r, s in zip(episode_rewards, episode_successes) if s > 0]
        if success_rewards:
            print(f"\n  --- Successful Episodes Only ---")
            print(f"  Count:                   {len(success_rewards)}")
            print(f"  Mean reward:             {np.mean(success_rewards):.2f} ± {np.std(success_rewards):.2f}")
            print(f"  Min reward:              {np.min(success_rewards):.2f}")
            print(f"  Max reward:              {np.max(success_rewards):.2f}")
        else:
            print(f"\n  No successful episodes.")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    args.test = True
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    evaluate()
