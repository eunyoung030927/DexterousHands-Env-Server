cd /data/eun_ws/DexterousHands-Env-Server/bidexhands

python train.py --task=ShadowHandGraspAndPlace --algo=ppo_collect --model_dir=./logs/ShadowHandGraspAndPlace/ppo/ppo_seed-1/model_1200.pt --test --num_envs=200
