"""
Isaac Gym Server for Bi-DexHands ShadowHandOver Environment
Serves environment observations and handles step requests via ZMQ
"""
import isaacgym
import torch
import numpy as np
import zmq
import pickle
import sys
import os
import traceback

# Change to DexterousHands directory to load configs properly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import Bi-DexHands utilities
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params
from bidexhands.utils.parse_task import parse_task

def get_server_args():
    """
    Configure command-line arguments for running Bi-DexHands
    as a server-side environment.
    """
    # task: ShadowHandOver 
    if "--task" not in sys.argv:
        sys.argv.append("--task")
        sys.argv.append("ShadowHandOver")
    
    # headless mode 
    # (Comment this out if you want to see rendering via VNC)
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")

    # Parse arguments
    args = get_args()
    
    # Training configuration (PPO)
    args.cfg_train = "bidexhands/cfg/ppo/config.yaml"
    
    # Environment configuration (ShadowHandOver Task)
    args.cfg_env = "bidexhands/cfg/ShadowHandOver.yaml"
    
    # Other settings
    args.num_envs = 1
    args.compute_device_id = 0
    args.graphics_device_id = 0
    
    print(f"Server Args configured:")
    print(f"  - Task: {args.task}")
    print(f"  - Config Train: {args.cfg_train}")
    print(f"  - Config Env: {args.cfg_env}")
    print(f"  - Num Envs: {args.num_envs}")
    print(f"  - Device: {args.device}")
    
    return args

def tensor_to_numpy(tensor):
    """
    Convert Isaac Gym GPU Tensor to serializable Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor

def run_server():
    """
    ZMQ Server that loads Bi-DexHands environment and handles client requests
    """
    server_socket = None
    context = None
    env = None
    
    try:
        # 1. Initialize ZMQ server (port 5555)
        print("🔌 Initializing ZMQ Server...")
        context = zmq.Context()
        server_socket = context.socket(zmq.REP)
        server_socket.bind("tcp://0.0.0.0:5555")
        print("ZMQ Server bound to tcp://0.0.0.0:5555")

        # 2. Load Bi-DexHands environment
        print(f"Loading Bi-DexHands Task: ShadowHandOver...")
        args = get_server_args()
        cfg, cfg_train, logdir = load_cfg(args)
        
        # Force single-environment setup
        cfg["env"]["numEnvs"] = 1
        cfg["env"]["episodeLength"] = 75  # Default episode length
        
        # Override asset root with an absolute path
        # "../assets" relative to bidexhands/cfg directory becomes /data/eun_ws/DexterousHands/assets
        assets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        cfg["env"]["asset"]["assetRoot"] = assets_root
        
        print(f"Config loaded")
        print(f" - Asset Root: {cfg['env']['asset']['assetRoot']}")
        print(f" - Observation Dim: {cfg['env'].get('obsSize', 'Not specified')}")
        print(f" - Action Dim: {cfg['env'].get('actionSize', 'Not specified')}")
        
        sim_params = parse_sim_params(args, cfg, cfg_train)
        print(f"✓ Sim params parsed")
        
        # Create task and environment 
        # parse_task returns (task, env) tuple
        print(f"Creating environment...")
        task, env = parse_task(args, cfg, cfg_train, sim_params, 0)
        print(f"Environment created successfully!")
        print(f" - Task Type: {type(task).__name__}")
        print(f" - Env Type: {type(env).__name__}")
        
        # Get device from environment
        device = env.rl_device if hasattr(env, 'rl_device') else torch.device('cuda:0')
        print(f"✓ Device: {device}")
        
        # Get observation and action space info from env
        if hasattr(env, 'observation_space'):
            obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None
        elif hasattr(env, 'num_obs'):
            obs_dim = env.num_obs
        else:
            obs_dim = None
            
        if hasattr(env, 'action_space'):
            action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else None
        elif hasattr(env, 'num_acts'):
            action_dim = env.num_acts
        else:
            action_dim = None
            
        print(f"✓ Observation Dim: {obs_dim}, Action Dim: {action_dim}")
        print(f"Server Ready! Waiting for Client on port 5555...")

        # 3. ZMQ request handling loop
        while True:
            try:
                # Wait for client request
                message = server_socket.recv()
                cmd, data = pickle.loads(message)
                response = None

                if cmd == "reset":
                    # --- RESET ---
                    print("[CMD] reset")
                    # VecTaskPython.reset() returns a Tensor directly, not a dict
                    obs_tensor = env.reset()
                    
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    
                    # Handle 2D case (num_envs, obs_dim) -> take first environment
                    if obs.ndim == 2:
                        obs = obs[0]
                    
                    response = ("ok", obs)
                    print(f"reset done, obs shape: {obs.shape}")

                elif cmd == "step":
                    # --- STEP ---
                    # Client sends action as NumPy array: (action_dim,)
                    # Isaac Gym expects (num_envs, action_dim)

                    current_mass_str = "N/A" # default value
                    try:
                        env_ptr = task.envs[0]
                        actor_handle = task.gym.find_actor_handle(env_ptr, "object")
                        if actor_handle != isaacgym.gymapi.INVALID_HANDLE:
                            props = task.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
                            if len(props) > 0:
                                current_mass_str = f"{props[0].mass:.4f} kg"
                    except:
                        pass # ignore errors in mass retrieval

                    action_array = np.array(data, dtype=np.float32)
                    action_tensor = torch.tensor(action_array, dtype=torch.float, device=device).unsqueeze(0)
                    
                    # Step the environment
                    obs_tensor, reward_tensor, done_tensor, info_dict = env.step(action_tensor)
                    
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    
                    if obs.ndim == 2:
                        obs = obs[0]
                    
                    rew = float(reward_tensor[0].item()) if hasattr(reward_tensor[0], 'item') else float(reward_tensor[0])
                    terminated = bool(done_tensor[0].item()) if hasattr(done_tensor[0], 'item') else bool(done_tensor[0])
                    truncated = False  # Isaac Gym does not clearly separate truncation
                    info = {}  # Convert tensors inside info_dict if needed

                    response = ("ok", (obs, rew, terminated, truncated, info, current_mass_str))
                    print(f"[CMD] step done | Mass: {current_mass_str} | Reward: {rew:.4f} | Obs: {obs.shape}")

                elif cmd == "close":
                    print("Server shutting down (client requested)...")
                    response = ("ok", None)
                    server_socket.send(pickle.dumps(response))
                    break
                
                else:
                    print(f"Unknown command: {cmd}")
                    response = ("error", f"Unknown command: {cmd}")

                # Send response back to client
                server_socket.send(pickle.dumps(response))

            except KeyboardInterrupt:
                print("\nServer interrupted by user")
                break
            except Exception as e:
                print(f"Error in communication loop: {e}")
                traceback.print_exc()
                try:
                    server_socket.send(pickle.dumps(("error", str(e))))
                except:
                    pass

    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        traceback.print_exc()
        
    finally:
        # Cleanup
        if server_socket is not None:
            server_socket.close()
        if context is not None:
            context.term()
        print("Server shutdown complete")

if __name__ == "__main__":
    run_server()