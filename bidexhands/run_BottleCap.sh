#!/bin/bash

# 에러가 발생해도 다음 실험으로 넘어가도록 설정 (중단 원하면 set -e 사용)
set +e 

echo "Start Training Sequence: MAPPO -> HAPPO -> PPO"

# 1. MAPPO 실행
python train_wandb.py --task=ShadowHandBottleCap --algo=mappo --headless
echo ">>> MAPPO Finished (or stopped)."
sleep 5

# 2. HAPPO 실행
python train_wandb.py --task=ShadowHandBottleCap --algo=happo --headless
echo ">>> HAPPO Finished (or stopped)."
sleep 5

echo "   All Experiments Completed."