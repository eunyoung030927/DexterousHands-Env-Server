"""
데이터셋 경로를 입력받아 rewards.npy + dones.npy로부터
에피소드별 total reward를 계산하고 결과 파일과 히스토그램을 저장하는 스크립트.

Usage:
    python compute_episode_rewards.py <dataset_path> [--name DATASET_NAME]

예시:
    python compute_episode_rewards.py logs/ShadowHandSwingCup/ppo/ppo_seed-1/ShadowHandSwingCup
    python compute_episode_rewards.py logs/ShadowHandSwingCup/ppo/ppo_seed-1/ShadowHandSwingCup --name my_dataset
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_episode_rewards(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """rewards, dones를 이용해 에피소드별 total reward를 계산."""
    rewards = rewards.flatten()
    dones = dones.flatten()

    done_indices = np.where(dones == 1.0)[0]
    if len(done_indices) == 0:
        print("[WARNING] done 신호가 없습니다. 전체를 하나의 에피소드로 처리합니다.")
        return np.array([rewards.sum()], dtype=np.float32)

    ep_rewards = []
    start = 0
    for end in done_indices:
        ep_rewards.append(rewards[start : end + 1].sum())
        start = end + 1

    # 마지막 done 이후 남은 스텝이 있으면 추가
    if start < len(rewards):
        ep_rewards.append(rewards[start:].sum())

    return np.array(ep_rewards, dtype=np.float32)


def save_histogram(episode_rewards: np.ndarray, dataset_name: str, save_path: str):
    """에피소드 reward 분포 히스토그램을 저장."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(episode_rewards, bins=50, edgecolor="black", alpha=0.7)
    mean_val = episode_rewards.mean()
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.set_title(f"{dataset_name} - Episode Reward Distribution ({len(episode_rewards)} episodes)")
    ax.set_xlabel("Episode Total Reward")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="에피소드별 total reward 계산 및 히스토그램 생성")
    parser.add_argument("dataset_path", help="rewards.npy, dones.npy가 있는 데이터셋 경로")
    parser.add_argument("--name", default=None, help="결과 저장 폴더 이름 (기본: 데이터셋 경로의 마지막 폴더명)")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset_path)

    # rewards.npy, dones.npy 확인
    rewards_path = os.path.join(dataset_path, "rewards.npy")
    dones_path = os.path.join(dataset_path, "dones.npy")

    if not os.path.isfile(rewards_path):
        print(f"[ERROR] {rewards_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)
    if not os.path.isfile(dones_path):
        print(f"[ERROR] {dones_path} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    # 기존 episode_rewards.npy가 있으면 그걸 사용, 없으면 계산
    episode_rewards_path = os.path.join(dataset_path, "episode_rewards.npy")
    if os.path.isfile(episode_rewards_path):
        print(f"[INFO] 기존 episode_rewards.npy 발견: {episode_rewards_path}")
        episode_rewards = np.load(episode_rewards_path)
    else:
        print("[INFO] episode_rewards.npy가 없어 rewards.npy + dones.npy로부터 계산합니다.")
        rewards = np.load(rewards_path)
        dones = np.load(dones_path)
        episode_rewards = compute_episode_rewards(rewards, dones)

    # 데이터셋 이름 결정
    dataset_name = args.name if args.name else os.path.basename(dataset_path)

    # 출력 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "run_logs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # episode_rewards.npy 저장
    out_npy = os.path.join(output_dir, "episode_rewards.npy")
    np.save(out_npy, episode_rewards)

    # episode_rewards.csv 저장 (읽기 편하도록)
    out_csv = os.path.join(output_dir, "episode_rewards.csv")
    with open(out_csv, "w") as f:
        f.write("episode,total_reward\n")
        for i, r in enumerate(episode_rewards):
            f.write(f"{i},{r:.4f}\n")

    # 히스토그램 저장
    out_png = os.path.join(output_dir, "episode_rewards_hist.png")
    save_histogram(episode_rewards, dataset_name, out_png)

    # 통계 출력
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Episodes: {len(episode_rewards)}")
    print(f"Mean Reward:   {episode_rewards.mean():.2f}")
    print(f"Std Reward:    {episode_rewards.std():.2f}")
    print(f"Min Reward:    {episode_rewards.min():.2f}")
    print(f"Max Reward:    {episode_rewards.max():.2f}")
    print(f"Median Reward: {np.median(episode_rewards):.2f}")
    print(f"{'='*50}")
    print(f"\n[SAVED] {out_npy}")
    print(f"[SAVED] {out_csv}")
    print(f"[SAVED] {out_png}")


if __name__ == "__main__":
    main()
