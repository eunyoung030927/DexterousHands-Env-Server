"""
ShadowHandSwingCup 테스트 + rot_dist 디버그 모니터링
원본 코드 수정 없이 compute_reward를 monkey-patch하여 rot_dist 값을 출력합니다.

Usage:
    cd /data/eun_ws/DexterousHands-Env-Server/bidexhands
    python test_swing_cup_debug.py --task=ShadowHandSwingCup --algo=ppo \
        --model_dir=logs/ShadowHandSwingCup/ppo/ppo_seed-1/model_1200.pt
"""

import isaacgym  # must be imported before torch

import numpy as np
import random
import math
import torch

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex
from bidexhands.algorithms.rl.ppo import PPO
from bidexhands.utils.torch_jit_utils import quat_mul, quat_conjugate

SUCCESS_THRESHOLD = 1  # radians (~45 degrees)

def patch_compute_reward(task):
    """Wrap compute_reward to print episode-level rot_dist/success/reward stats"""
    original_compute_reward = task.compute_reward

    num_envs = task.num_envs
    device = task.device
    step_count = [0]

    # per-env episode trackers
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
    cur_min_rot_dist = torch.full((num_envs,), float('inf'), dtype=torch.float, device=device)
    cur_ever_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # completed episode stats
    episode_rewards = []
    episode_min_rot_dists = []
    episode_successes = []  # 1 if ever hit threshold during episode

    def debug_compute_reward(actions):
        nonlocal cur_reward_sum, cur_min_rot_dist, cur_ever_success

        # original reward computation
        original_compute_reward(actions)

        # compute rot_dist
        quat_diff = quat_mul(task.object_rot, quat_conjugate(task.goal_rot))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        # update per-env episode trackers
        cur_reward_sum += task.rew_buf
        cur_min_rot_dist = torch.minimum(cur_min_rot_dist, rot_dist)
        cur_ever_success = cur_ever_success | (rot_dist < SUCCESS_THRESHOLD)

        # collect finished episodes
        dones = task.reset_buf
        done_mask = dones > 0
        if done_mask.any():
            episode_rewards.extend(cur_reward_sum[done_mask].cpu().tolist())
            episode_min_rot_dists.extend(cur_min_rot_dist[done_mask].cpu().tolist())
            episode_successes.extend(cur_ever_success[done_mask].cpu().tolist())

            # reset trackers for done envs
            cur_reward_sum[done_mask] = 0.0
            cur_min_rot_dist[done_mask] = float('inf')
            cur_ever_success[done_mask] = False

        step_count[0] += 1

        if step_count[0] % 50 == 0:
            n_eps = len(episode_rewards)
            if n_eps > 0:
                recent_n = min(100, n_eps)
                recent_rewards = episode_rewards[-recent_n:]
                recent_rot = episode_min_rot_dists[-recent_n:]
                recent_succ = episode_successes[-recent_n:]

                mean_reward = sum(recent_rewards) / recent_n
                mean_min_rot = sum(recent_rot) / recent_n
                min_min_rot = min(recent_rot)
                success_rate = sum(recent_succ) / recent_n

                print(f"\n[Step {step_count[0]:>6d}] ======== Episode Stats (last {recent_n} / total {n_eps}) ========")
                print(f"  mean reward         | {mean_reward:.2f}")
                print(f"  min rot_dist (rad)  | mean: {mean_min_rot:.4f}  best: {min_min_rot:.4f}")
                print(f"  min rot_dist (deg)  | mean: {mean_min_rot*180/math.pi:.1f}   best: {min_min_rot*180/math.pi:.1f}")
                print(f"  success thresh      | {SUCCESS_THRESHOLD} rad ({SUCCESS_THRESHOLD*180/math.pi:.1f} deg)")
                print(f"  success rate        | {sum(recent_succ)}/{recent_n} ({100*success_rate:.1f}%)")
                print(f"  ================================================================")
            else:
                print(f"\n[Step {step_count[0]:>6d}] No episodes completed yet")

    task.compute_reward = debug_compute_reward


def main():
    set_np_formatting()
    args = get_args()
    args.test = True  # 강제 테스트 모드

    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    # monkey-patch로 디버그 출력 추가
    patch_compute_reward(task)

    learn_cfg = cfg_train["learn"]
    logdir = logdir + "_seed{}".format(env.task.cfg["seed"])

    model = PPO(
        vec_env=env,
        cfg_train=cfg_train,
        device=env.rl_device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=True,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        asymmetric=(env.num_states > 0),
    )

    chkpt_path = args.model_dir
    print(f"\n=== Loading model from {chkpt_path} ===")
    model.test(chkpt_path)

    print(f"\n=== Test started (success threshold: {SUCCESS_THRESHOLD} rad / {SUCCESS_THRESHOLD*180/math.pi:.1f} deg) ===\n")
    model.run(num_learning_iterations=0, log_interval=1)


if __name__ == "__main__":
    main()
