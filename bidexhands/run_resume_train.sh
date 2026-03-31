cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

# =============================================================
# Domain Randomization Resume Training
# 기존 체크포인트에서 이어서 학습 (config에서 randomize 설정)
# =============================================================

# Batch 1: LiftUnderarm + DoorOpenOutward 병렬
echo "[Batch 1] Starting DR: ShadowHandLiftUnderarm, ShadowHandDoorOpenOutward"
python train_wandb.py --task=ShadowHandLiftUnderarm --algo=ppo \
  --model_dir=logs/ShadowHandLiftUnderarm/ppo/ppo_seed-1/model_6501.pt \
  --max_iterations=10000 --headless \
  > $LOG_DIR/resume_dr_ShadowHandLiftUnderarm.log 2>&1 &

python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=ppo \
  --model_dir=logs/ShadowHandDoorOpenOutward/ppo/ppo_seed-1/model_1200.pt \
  --max_iterations=2000 --headless \
  > $LOG_DIR/resume_dr_ShadowHandDoorOpenOutward.log 2>&1 &
wait
echo "[Batch 1] Done"

# Batch 2: SwingCup + GraspAndPlace 병렬
echo "[Batch 2] Starting DR: ShadowHandSwingCup, ShadowHandGraspAndPlace"
python train_wandb.py --task=ShadowHandSwingCup --algo=ppo \
  --model_dir=logs/ShadowHandSwingCup/ppo/ppo_seed-1/model_1200.pt \
  --max_iterations=2000 --headless \
  > $LOG_DIR/resume_dr_ShadowHandSwingCup.log 2>&1 &

python train_wandb.py --task=ShadowHandGraspAndPlace --algo=ppo \
  --model_dir=logs/ShadowHandGraspAndPlace/ppo/ppo_seed-1/model_1200.pt \
  --max_iterations=2000 --headless \
  > $LOG_DIR/resume_dr_ShadowHandGraspAndPlace.log 2>&1 &
wait
echo "[Batch 2] Done"

echo "All DR resume training complete"
