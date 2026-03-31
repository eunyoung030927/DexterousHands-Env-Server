cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

# Batch 1: 2개 병렬 실행
echo "[Batch 1] Starting: ShadowHandLiftUnderarm, ShadowHandDoorOpenOutward"
python train_wandb.py --task=ShadowHandLiftUnderarm --algo=ppo --headless > $LOG_DIR/train_ShadowHandLiftUnderarm.log 2>&1 &
python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=ppo --headless > $LOG_DIR/train_ShadowHandDoorOpenOutward.log 2>&1 &
wait
echo "[Batch 1] Done"

# # Batch 2: 2개 병렬 실행
# echo "[Batch 2] Starting: ShadowHandSwingCup, ShadowHandGraspAndPlace"
# python train_wandb.py --task=ShadowHandSwingCup --algo=ppo --headless --max_iterations=1200 > $LOG_DIR/train_ShadowHandSwingCup.log 2>&1 &
# python train_wandb.py --task=ShadowHandGraspAndPlace --algo=ppo --headless --max_iterations=1200 > $LOG_DIR/train_ShadowHandGraspAndPlace.log 2>&1 &
# wait
# echo "[Batch 2] Done"
# echo "All training complete"