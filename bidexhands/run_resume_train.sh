cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

# ShadowHandLiftUnderarm: 리워드 아직 상승 중, 추가학습 유효
python train_wandb.py --task=ShadowHandLiftUnderarm --algo=ppo --model_dir=logs/ShadowHandLiftUnderarm/ppo/ppo_seed-1/model_6501.pt --max_iterations=13000 --headless > $LOG_DIR/resume_ShadowHandLiftUnderarm.log 2>&1

echo "All resume training complete"
