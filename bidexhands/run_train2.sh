cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

# ShadowHandDoorOpenOutward: MAPPO로 재학습 (양손 협응에 유리)
# ShadowHandSwingCup: PPO 재학습 (LR 1e-4, entropy 0.001로 안정화)
echo "[Batch 1] Starting: ShadowHandDoorOpenOutward (MAPPO), ShadowHandSwingCup (PPO)"
python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=mappo --headless > $LOG_DIR/train2_ShadowHandDoorOpenOutward_mappo.log 2>&1 &
python train_wandb.py --task=ShadowHandSwingCup --algo=ppo --cfg_train=cfg/ppo/swing_cup_config.yaml --headless > $LOG_DIR/train2_ShadowHandSwingCup.log 2>&1 &
wait
echo "[Batch 1] Done"

echo "All training complete"
