cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

python train_wandb.py --task=ShadowHandLiftUnderarm --algo=ppo --headless 

python train_wandb.py --task=ShadowHandDoorOpenOutward --algo=ppo --headless --max_iterations=1200

python train_wandb.py --task=ShadowHandSwingCup --algo=ppo --headless --max_iterations=1200

python train_wandb.py --task=ShadowHandGraspAndPlace --algo=ppo --headless --max_iterations=1200