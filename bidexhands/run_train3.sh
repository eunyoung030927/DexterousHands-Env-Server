cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

LOG_DIR=/data/eun_ws/DexterousHands-Env-Server/bidexhands/run_logs
mkdir -p $LOG_DIR

echo "Starting"
# python train_wandb.py --task=ShadowHandScissors --algo=ppo --headless > $LOG_DIR/train_ShadowHandScissors.log 2>&1 
# python train_wandb.py --task=ShadowHandScissors --algo=ppo --headless --num_envs=1024 --cfg_train=cfg/ppo/scissors_config.yaml 2>&1 | tee $LOG_DIR/train_ShadowHandScissors2.log

# python train_wandb.py --task=ShadowHandPen --algo=ppo --headless > $LOG_DIR/train_ShadowHandPen.log 2>&1 
# python train_wandb.py --task=ShadowHandPen --algo=ppo --headless --num_envs=1024 --cfg_train=cfg/ppo/scissors_config.yaml 2>&1 | tee $LOG_DIR/train_ShadowHandPen2.log

python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=ppo --headless --num_envs=1024 --cfg_train=cfg/ppo/scissors_config.yaml 2>&1 | tee $LOG_DIR/train_ShadowHandDoorOpenOutward2.log
echo "Done"
