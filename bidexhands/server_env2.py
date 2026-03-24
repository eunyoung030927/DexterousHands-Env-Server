"""
Isaac Gym Server for Bi-DexHands
Serves environment observations and handles step requests via ZMQ
Dynamically loads config based on --task and --algo arguments
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
from bidexhands.utils.config import get_args, load_cfg, parse_sim_params, set_np_formatting, set_seed
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex

# Algorithm Lists (Same as train.py)
MARL_ALGOS = ["mappo", "happo", "hatrpo", "maddpg", "ippo"]
SARL_ALGOS = ["ppo", "ddpg", "sac", "td3", "trpo"]
MTRL_ALGOS = ["mtppo", "random"]
META_ALGOS = ["mamlppo"]
OFFRL_ALGOS = ["td3_bc", "bcq", "iql", "ppo_collect"]

def configure_server_args():
    """
    Parse arguments and setup task type based on algorithm
    """
    # Parse arguments using Bi-DexHands utility
    # This handles --task, --algo, --headless, etc.
    args = get_args()

    # Default logic for headless mode if not specified (Server usually runs headless)
    # If you want to force headless unless specified otherwise:
    if args.headless is False and "--headless" not in sys.argv:
         # Note: get_args() sets headless=False by default if not provided. 
         # If you want to enforce headless for server, uncomment below:
         args.headless = True
         pass

    print(f"Server Args initialized:")
    print(f"  - Task: {args.task}")
    print(f"  - Algo: {args.algo}")
    
    # --- Logic from train.py to determine task type ---
    if args.algo in MARL_ALGOS:
        args.task_type = "MultiAgent"
    elif args.algo in MTRL_ALGOS:
        args.task_type = "MultiTask"
    elif args.algo in META_ALGOS:
        args.task_type = "Meta"
    elif args.algo in SARL_ALGOS:
        args.task_type = "SingleAgent" # Usually handled implicitly, but good to be explicit
    elif args.algo in OFFRL_ALGOS:
        pass # Offline RL specific handling if needed
    else:
        print(f"Warning: Algorithm {args.algo} not recognized in standard lists. Assuming standard setup.")

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
        set_np_formatting()
        args = configure_server_args()
        
        # Load Configs (Automatically finds path based on args.task and args.algo)
        cfg, cfg_train, logdir = load_cfg(args)
        
        # --- Force Server-Specific Environment Settings ---
        # Force single-environment setup for the server
        cfg["env"]["numEnvs"] = 1
        # Default episode length if not set (can be overridden by config)
        if "episodeLength" not in cfg["env"]:
            cfg["env"]["episodeLength"] = 75

        # Override asset root with an absolute path to avoid relative path issues
        assets_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
        cfg["env"]["asset"]["assetRoot"] = assets_root
        
        print(f"Config loaded dynamically:")
        print(f" - Asset Root: {cfg['env']['asset']['assetRoot']}")
        print(f" - Task Type: {getattr(args, 'task_type', 'Default')}")
        
        # Parse Simulation Parameters
        sim_params = parse_sim_params(args, cfg, cfg_train)
        
        # Set Seed
        set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
        
        # Get Agent Index (Needed for some MARL tasks)
        agent_index = get_AgentIndex(cfg)

        # Create task and environment 
        print(f"Creating environment...")
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        
        print(f"Environment created successfully!")
        print(f" - Env Class: {type(env).__name__}")
        
        # Get device
        device = env.rl_device if hasattr(env, 'rl_device') else torch.device('cuda:0')
        print(f"✓ Device: {device}")
        
        # Display Obs/Action Info
        obs_dim = None
        action_dim = None
        
        if hasattr(env, 'observation_space'):
            obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None
        elif hasattr(env, 'num_obs'):
            obs_dim = env.num_obs
            
        if hasattr(env, 'action_space'):
            action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else None
        elif hasattr(env, 'num_acts'):
            action_dim = env.num_acts
            
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
                    obs_tensor = env.reset()
                    
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    
                    # Handle (num_envs, obs_dim) -> take first environment
                    if obs.ndim == 2:
                        obs = obs[0]
                    
                    response = ("ok", obs)
                    print(f"reset done, obs shape: {obs.shape}")

                elif cmd == "step":
                    # --- STEP ---
                    # Client sends action as NumPy array: (action_dim,)
                    
                    current_mass_str = "N/A"
                    # Try to get mass info (specific to object handling tasks)
                    try:
                        # Assuming single env at index 0
                        env_ptr = task.envs[0] 
                        # Note: "object" might not exist in all tasks, wrap in try/except
                        actor_handle = task.gym.find_actor_handle(env_ptr, "object")
                        if actor_handle != isaacgym.gymapi.INVALID_HANDLE:
                            props = task.gym.get_actor_rigid_body_properties(env_ptr, actor_handle)
                            if len(props) > 0:
                                current_mass_str = f"{props[0].mass:.4f} kg"
                    except:
                        pass 

                    action_array = np.array(data, dtype=np.float32)
                    
                    # Ensure action shape matches what Isaac Gym expects (num_envs, action_dim)
                    # For MARL, the action shape handling might need adjustment based on the specific env implementation
                    if args.task_type == "MultiAgent":
                        # MARL tasks usually expect combined actions or specific shaping
                        # This is a naive implementation assuming the client sends the correct flattened/combined action
                        action_tensor = torch.tensor(action_array, dtype=torch.float, device=device)
                        if action_tensor.ndim == 1:
                            action_tensor = action_tensor.unsqueeze(0)
                    else:
                        action_tensor = torch.tensor(action_array, dtype=torch.float, device=device).unsqueeze(0)
                    
                    # Step the environment
                    obs_tensor, reward_tensor, done_tensor, info_dict = env.step(action_tensor)
                    
                    # Process Obs
                    if isinstance(obs_tensor, torch.Tensor):
                        obs = obs_tensor.cpu().detach().numpy()
                    else:
                        obs = np.array(obs_tensor)
                    if obs.ndim == 2:
                        obs = obs[0]
                    
                    # Process Reward
                    if isinstance(reward_tensor, torch.Tensor):
                        rew = float(reward_tensor[0].item())
                    else:
                        rew = float(reward_tensor[0])
                        
                    # Process Done
                    if isinstance(done_tensor, torch.Tensor):
                        terminated = bool(done_tensor[0].item())
                    else:
                        terminated = bool(done_tensor[0])
                        
                    truncated = False 
                    info = {} 

                    response = ("ok", (obs, rew, terminated, truncated, info, current_mass_str))
                    print(f"[CMD] step | Mass: {current_mass_str} | R: {rew:.4f} | Done: {terminated}")

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
        if server_socket is not None:
            server_socket.close()
        if context is not None:
            context.term()
        print("Server shutdown complete")

if __name__ == "__main__":
    run_server()