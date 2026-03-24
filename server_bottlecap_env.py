"""
Isaac Gym Server for Bi-DexHands ShadowHandBottleCap Environment
(Modified for Concatenated Observation: 442 dims)
"""
import isaacgym
import torch
import numpy as np
import zmq
import pickle
import sys
import os
import traceback

# ---------------------------------------------------------
# [FIX] Force Change Working Directory
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "bidexhands")

if os.path.exists(target_dir):
    print(f"📂 Changing working directory to: {target_dir}")
    os.chdir(target_dir)
    sys.path.append(script_dir)
else:
    print(f"📂 Current working directory: {os.getcwd()}")
    sys.path.append(os.getcwd())

from bidexhands.utils.config import get_args, load_cfg, parse_sim_params
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex

def get_server_args():
    if "--task" not in sys.argv:
        sys.argv.append("--task")
        sys.argv.append("ShadowHandBottleCap")
    
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")

    args = get_args()
    args.cfg_train = "cfg/ppo/config.yaml"
    args.cfg_env = "cfg/ShadowHandBottleCap_server.yaml"
    
    # [핵심] MultiAgent 환경으로 설정해야 (num_envs, 2, 221) shape 반환
    args.task_type = "MultiAgent"
    
    args.num_envs = 1
    args.compute_device_id = 0
    args.graphics_device_id = 0
    
    print(f"Server Args configured: {args.task}, task_type: {args.task_type}")
    return args

def run_server():
    server_socket = None
    context = None
    env = None
    
    try:
        print("🔌 Initializing ZMQ Server...")
        context = zmq.Context()
        server_socket = context.socket(zmq.REP)
        server_socket.bind("tcp://0.0.0.0:5555")
        print("ZMQ Server bound to tcp://0.0.0.0:5555")

        print(f"Loading Bi-DexHands Task: ShadowHandBottleCap...")
        args = get_server_args()
        cfg, cfg_train, logdir = load_cfg(args)
        
        cfg["env"]["numEnvs"] = 1
        cfg["env"]["episodeLength"] = 125  

        cfg["env"]["asymmetric_observations"] = True
        
        current_work_dir = os.getcwd()
        project_root = os.path.dirname(current_work_dir)
        assets_root = os.path.join(project_root, "assets")
        
        if os.path.exists(assets_root):
             print(f"✅ Found assets at: {assets_root}")
             cfg["env"]["asset"]["assetRoot"] = assets_root
        
        sim_params = parse_sim_params(args, cfg, cfg_train)
        
        # MultiAgent 환경을 위한 agent_index 설정
        agent_index = get_AgentIndex(cfg)
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        
        device = env.rl_device if hasattr(env, 'rl_device') else torch.device('cuda:0')
        print(f"✓ Device: {device}")
        print(f"Server Ready! Expecting/Sending Flattened Data (Obs: 442, Act: 52)...")

        while True:
            try:
                message = server_socket.recv()
                cmd, data = pickle.loads(message)
                response = None

                if cmd == "reset":
                    print("[CMD] reset")
                    # MultiVecTaskPython.reset() returns (obs, state_all, None)
                    # obs shape: (num_envs, num_agents, obs_dim) = (1, 2, 221)
                    reset_result = env.reset()
                    obs_tensor = reset_result[0]  # 첫 번째가 obs
                    
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    
                    # [핵심] (1, 2, 221) -> (442,) flatten
                    obs = obs.flatten() 
                    
                    response = ("ok", obs)
                    print(f"reset done, flattened obs shape: {obs.shape}") # (442,) 확인

                elif cmd == "step":
                    # data: Client가 보낸 Action (52,) - 2 agents * 26 actions
                    
                    current_mass_str = "N/A"
                    try:
                        env_ptr = task.envs[0]
                        actor_handle = task.gym.find_actor_handle(env_ptr, "object")
                        if actor_handle != isaacgym.gymapi.INVALID_HANDLE:
                            props = task.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
                            if len(props) > 0: current_mass_str = f"{props[0].mass:.4f} kg"
                    except: pass

                    # [핵심] MultiVecTaskPython.step()은 actions를 리스트로 받음
                    # Client(52,) -> [agent0_action(1,26), agent1_action(1,26)]
                    action_array = np.array(data, dtype=np.float32).flatten()
                    
                    # 2 agents, 각각 26 actions
                    action0 = torch.tensor(action_array[:26], dtype=torch.float, device=device).unsqueeze(0)  # (1, 26)
                    action1 = torch.tensor(action_array[26:], dtype=torch.float, device=device).unsqueeze(0)  # (1, 26)
                    actions_list = [action0, action1]
                    
                    # MultiVecTaskPython.step() returns (obs_all, state_all, reward_all, done_all, info_all, None)
                    obs_tensor, state_tensor, reward_tensor, done_tensor, info_tensor, _ = env.step(actions_list)
                    
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    
                    # [핵심] (1, 2, 221) -> (442,) flatten
                    obs = obs.flatten()
                    
                    # reward_all shape: (num_envs, num_agents, 1) = (1, 2, 1)
                    # 두 agent의 reward 평균 또는 합
                    if isinstance(reward_tensor, torch.Tensor):
                        rew = float(reward_tensor.mean().item())
                    else:
                        rew = float(np.mean(reward_tensor))
                    
                    # done_all shape: (num_envs, num_agents) = (1, 2)
                    # 둘 중 하나라도 done이면 terminated
                    if isinstance(done_tensor, torch.Tensor):
                        terminated = bool(done_tensor.any().item())
                    else:
                        terminated = bool(np.any(done_tensor))
                    truncated = False
                    
                    info = {}
                    if isinstance(info_tensor, dict):
                        for k, v in info_tensor.items():
                            if isinstance(v, torch.Tensor): info[k] = v.cpu().detach().numpy().tolist()
                            else: info[k] = v
                    
                    # [핵심 수정 1] Mass 정보를 info 딕셔너리에 넣습니다.
                    info["mass"] = current_mass_str 

                    # [핵심 수정 2] 반환값 튜플을 5개로 줄입니다! (current_mass_str 제거)
                    # 수정 전: response = ("ok", (obs, rew, terminated, truncated, info, current_mass_str))
                    # 수정 후:
                    response = ("ok", (obs, rew, terminated, truncated, info))

                elif cmd == "close":
                    print("Server shutting down...")
                    response = ("ok", None)
                    server_socket.send(pickle.dumps(response))
                    break
                else:
                    response = ("error", f"Unknown command: {cmd}")

                server_socket.send(pickle.dumps(response))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                try: server_socket.send(pickle.dumps(("error", str(e))))
                except: pass

    except Exception as e:
        print(f"Fatal Error: {e}")
        traceback.print_exc()
    finally:
        if server_socket: server_socket.close()
        if context: context.term()

if __name__ == "__main__":
    run_server()