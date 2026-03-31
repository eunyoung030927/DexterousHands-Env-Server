cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

python train.py --task=ShadowHandLiftUnderarm --algo=ppo --model_dir=logs/ShadowHandLiftUnderarm/ppo/ppo_seed-1/model_12000.pt --test

python train.py --task=ShadowHandDoorOpenOutward --algo=ppo --model_dir=logs/ShadowHandDoorOpenOutward/ppo/ppo_seed-1/model_1200.pt --test

python train.py --task=ShadowHandSwingCup --algo=ppo --model_dir=logs/ShadowHandSwingCup/ppo/ppo_seed-1/model_1200.pt --test

# python train.py --task=ShadowHandGraspAndPlace --algo=ppo --model_dir=logs/ShadowHandGraspAndPlace/ppo/ppo_seed-1/model_1200.pt --test