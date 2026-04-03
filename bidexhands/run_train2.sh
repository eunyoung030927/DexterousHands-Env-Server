cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

echo "[Batch 1] Starting: ShadowHandLiftUnderarm, ShadowHandDoorOpenOutward"
python train_wandb.py --task=ShadowHandLiftUnderarm --algo=ppo --headless > $LOG_DIR/train_ShadowHandLiftUnderarm.log 2>&1 &

python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=ppo \
  --model_dir=logs/ShadowHandDoorOpenOutward/ppo/ppo_seed-1/model_1200.pt \
  --max_iterations=2000 --headless --randomize \
  > $LOG_DIR/resume_dr_ShadowHandDoorOpenOutward.log 2>&1 &
wait

wait
echo "[Batch 1] Done"