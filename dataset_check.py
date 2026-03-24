# /data/eun_ws/DexterousHands/bidexhands/logs/exp2/shadow_hand_bottle_cap/mappo/models_seed1/mappo_dataset

import os
import numpy as np


path_to_dataset = f"/data/eun_ws/DexterousHands/bidexhands/logs/exp2/shadow_hand_bottle_cap/mappo/models_seed1/mappo_dataset"
# print(np.load(os.path.join(path_to_dataset, "states.npy")))


observations = np.load(os.path.join(path_to_dataset, "states.npy"))
next_observations = np.load(os.path.join(path_to_dataset, "next_states.npy"))
actions = np.load(os.path.join(path_to_dataset, "actions.npy"))
rewards = np.load(os.path.join(path_to_dataset, "rewards.npy"))
dones = np.load(os.path.join(path_to_dataset, "dones.npy"))
# successes = np.load(os.path.join(path_to_dataset, "successes.npy"))

print(f"📊 Observations shape : {observations.shape}")
print(f"📊 Next Obs shape     : {next_observations.shape}")
print(f"📊 Actions shape      : {actions.shape}")
print(f"📊 Rewards shape      : {rewards.shape}")
print(f"📊 Dones shape        : {dones.shape}")
# print(f"📊 Successes shape    : {successes.shape}")

# 데이터 정합성 체크
assert len(observations) == len(actions) == len(dones), \
    "Dataset arrays must have the same length"

print("✅ Dataset length check: SUCCESS")
