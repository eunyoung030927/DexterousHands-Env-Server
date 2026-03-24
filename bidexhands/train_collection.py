# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
import numpy as np
import random
import os

from bidexhands.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_sarl import process_sarl
from bidexhands.utils.process_marl import process_MultiAgentRL, get_AgentIndex
from bidexhands.utils.process_mtrl import *
from bidexhands.utils.process_metarl import *
from bidexhands.utils.process_offrl import *

import torch

# 알고리즘 목록 정의
MARL_ALGOS = ["mappo", "happo", "hatrpo", "maddpg", "ippo"]
SARL_ALGOS = ["ppo", "ddpg", "sac", "td3", "trpo"]
MTRL_ALGOS = ["mtppo", "random"]
META_ALGOS = ["mamlppo"]
OFFRL_ALGOS = ["td3_bc", "bcq", "iql", "ppo_collect"]

def train_collection():
    print("Algorithm: ", args.algo)
    agent_index = get_AgentIndex(cfg)

    # 알고리즘 확인
    assert args.algo in MARL_ALGOS + SARL_ALGOS + MTRL_ALGOS + META_ALGOS + OFFRL_ALGOS, \
        "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo, ippo, ...]"

    algo = args.algo
    
    # 1. Multi-Agent 알고리즘인 경우 (MAPPO 등)
    if args.algo in MARL_ALGOS:
        args.task_type = "MultiAgent"
        algo = "MultiAgentRL"
        
        # 환경 및 태스크 설정 로드
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        
        # Runner 초기화 (이때 model_dir이 있으면 내부적으로 restore()가 호출되어 모델이 로드됨)
        runner = eval('process_{}'.format(algo))(args, env, cfg_train, args.model_dir)
        
        # [핵심 수정 부분] 모델 경로가 있으면 학습(run)이나 평가(eval) 대신 데이터 수집(collect_data) 실행
        if args.model_dir != "":
            print("======================================================")
            print(f"Load model successful from: {args.model_dir}")
            print("Start Data Collection Mode...")
            print("======================================================")
            
            # 수집할 에피소드 수 설정 (필요시 args로 받아도 됨)
            COLLECT_EPISODES = 2048  
            SAVE_NAME = "mappo_dataset"
            
            # runner.py에 추가한 collect_data 함수 호출
            runner.collect_data(total_episodes=COLLECT_EPISODES, save_name=SAVE_NAME)
        else:
            print("[Error] --model_dir is required for data collection script!")
            print("Please provide the path to the trained model folder.")
            return
            
        return

    # 2. Single-Agent 알고리즘인 경우
    elif args.algo in SARL_ALGOS:
        algo = "sarl"
    elif args.algo in MTRL_ALGOS:
        args.task_type = "MultiTask"
    elif args.algo in META_ALGOS:
        args.task_type = "Meta"
    elif args.algo in OFFRL_ALGOS:
        pass

    # (참고) 현재 데이터 수집 로직은 MARL(MAPPO) 위주로 작성됨.
    # SARL 등 다른 알고리즘에 대해서도 수집하려면 해당 Runner에도 collect_data 함수가 있어야 함.
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    runner = eval('process_{}'.format(algo))(args, env, cfg_train, logdir)
    
    # SARL 등의 경우에도 model_dir 체크 후 수집 로직 연결 가능
    # 여기서는 MARL(MAPPO) 기준으로 작성되었으므로 생략하거나 동일하게 분기 처리 필요

if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    print("🔥🔥 Force Override: asymmetric_observations = True 🔥🔥")
    cfg["env"]["asymmetric_observations"] = True  # 강제 주입

    sim_params = parse_sim_params(args, cfg, cfg_train)
    
    # 시드 설정
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    # 데이터 수집 실행
    train_collection()