"""
Unified ZMQ Server for Bi-DexHands IsaacGym Environments

Supports both SingleAgent and MultiAgent tasks via --task flag.

Usage:
    python bidexhands_server_env.py --task ShadowHandCatchOver2Underarm
    python bidexhands_server_env.py --task ShadowHandOver
    python bidexhands_server_env.py --task ShadowHandBottleCap --task_type MultiAgent
"""
import isaacgym
import torch
import numpy as np
import zmq
import pickle
import io
import sys
import os
import traceback


# ---------------------------------------------------------------------------
# Compatibility shim: allow unpickling numpy 2.x arrays on numpy 1.x
# numpy 2.0 renamed numpy.core → numpy._core; when a client serialises an
# array with numpy 2.x, the pickle stream references numpy._core which does
# not exist in numpy 1.x.  This custom Unpickler transparently redirects
# those imports so that the server can deserialise without error.
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def compat_loads(data: bytes):
    return _NumpyCompatUnpickler(io.BytesIO(data)).load()


# ---------------------------------------------------------------------------
# Extract server-only args BEFORE IsaacGym's argparse sees sys.argv
# (gymutil.parse_arguments uses parse_args() which rejects unknown flags)
# ---------------------------------------------------------------------------
def _pop_argv(flag, default=None):
    """Remove --flag <value> pair from sys.argv and return the value."""
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        sys.argv.pop(idx)
        if idx < len(sys.argv) and not sys.argv[idx].startswith("--"):
            return sys.argv.pop(idx)
    return default


SERVER_HOST = _pop_argv("--host", "0.0.0.0")
SERVER_PORT = int(_pop_argv("--port", "5555"))


# ---------------------------------------------------------------------------
# Working directory: bidexhands/ (where cfg/ lives, matching retrieve_cfg paths)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIDEXHANDS_DIR = os.path.join(SCRIPT_DIR, "bidexhands")
os.chdir(BIDEXHANDS_DIR)
sys.path.append(SCRIPT_DIR)

from bidexhands.utils.config import get_args, load_cfg, parse_sim_params
from bidexhands.utils.parse_task import parse_task
from bidexhands.utils.process_marl import get_AgentIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_object_mass(task) -> str:
    """Query current object mass from the IsaacGym simulation."""
    try:
        env_ptr = task.envs[0]
        handle = task.gym.find_actor_handle(env_ptr, "object")
        if handle != isaacgym.gymapi.INVALID_HANDLE:
            props = task.gym.get_actor_rigid_body_properties(env_ptr, handle)
            if props:
                return f"{props[0].mass:.4f} kg"
    except:
        pass
    return "N/A"


def _to_numpy(t):
    if isinstance(t, torch.Tensor):
        return t.cpu().detach().numpy()
    return np.array(t)


def _convert_info(info_raw):
    """Convert info dict values from tensors to serializable types."""
    info = {}
    if isinstance(info_raw, dict):
        for k, v in info_raw.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.cpu().detach().numpy().tolist()
            else:
                info[k] = v
    return info


# ---------------------------------------------------------------------------
# EnvAdapter — absorbs SingleAgent / MultiAgent differences
# ---------------------------------------------------------------------------
class EnvAdapter:
    """Unified interface over VecTaskPython (single) and MultiVecTaskPython (multi).

    - reset()  → 1-D numpy obs
    - step(action_flat) → (obs, reward, terminated, truncated, info)
    """

    def __init__(self, env, task, task_type, device):
        self.env = env
        self.task = task
        self.is_multi_agent = (task_type == "MultiAgent")
        self.num_agents = env.num_agents
        self.action_dim_per_agent = env.num_acts
        self.device = device

    # ----- reset ----------------------------------------------------------
    def reset(self) -> np.ndarray:
        if self.is_multi_agent:
            # MultiVecTaskPython.reset() → (obs, state_all, None)
            # obs shape: (num_envs, num_agents, obs_dim) e.g. (1, 2, 221)
            result = self.env.reset()
            obs = _to_numpy(result[0]).flatten()
        else:
            # VecTaskPython.reset() → tensor (num_envs, obs_dim)
            obs = _to_numpy(self.env.reset())
            if obs.ndim == 2:
                obs = obs[0]
        return obs

    # ----- step -----------------------------------------------------------
    def step(self, action_flat: np.ndarray):
        """
        Args:
            action_flat: 1-D numpy array (all agents concatenated for multi-agent).
        Returns:
            (obs, reward, terminated, truncated, info)  — all scalars / 1-D numpy
        """
        if self.is_multi_agent:
            return self._step_multi(action_flat)
        return self._step_single(action_flat)

    def _step_single(self, action_flat):
        action_tensor = torch.tensor(
            np.array(action_flat, dtype=np.float32),
            dtype=torch.float, device=self.device,
        ).unsqueeze(0)  # (act_dim,) → (1, act_dim)

        # VecTaskPython.step() → (obs, reward, done, info)
        obs_t, rew_t, done_t, info_raw = self.env.step(action_tensor)

        obs = _to_numpy(obs_t)
        if obs.ndim == 2:
            obs = obs[0]

        rew = float(rew_t[0].item() if hasattr(rew_t[0], "item") else rew_t[0])
        terminated = bool(done_t[0].item() if hasattr(done_t[0], "item") else done_t[0])
        truncated = False

        info = _convert_info(info_raw)
        info["mass"] = get_object_mass(self.task)
        return obs, rew, terminated, truncated, info

    def _step_multi(self, action_flat):
        arr = np.array(action_flat, dtype=np.float32).flatten()
        dim = self.action_dim_per_agent

        # Split flat action into per-agent tensors: [tensor(1,26), tensor(1,26)]
        actions_list = []
        for i in range(self.num_agents):
            a = torch.tensor(
                arr[i * dim : (i + 1) * dim],
                dtype=torch.float, device=self.device,
            ).unsqueeze(0)
            actions_list.append(a)

        # MultiVecTaskPython.step() → (obs, state, reward, done, info, None)
        obs_t, _state, rew_t, done_t, info_raw, _ = self.env.step(actions_list)

        obs = _to_numpy(obs_t).flatten()

        if isinstance(rew_t, torch.Tensor):
            rew = float(rew_t.mean().item())
        else:
            rew = float(np.mean(rew_t))

        if isinstance(done_t, torch.Tensor):
            terminated = bool(done_t.any().item())
        else:
            terminated = bool(np.any(done_t))
        truncated = False

        info = _convert_info(info_raw)
        info["mass"] = get_object_mass(self.task)
        return obs, rew, terminated, truncated, info


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def run_server():
    # --- Determine task from sys.argv (before get_args consumes it) --------
    task_name = None
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        if idx + 1 < len(sys.argv):
            task_name = sys.argv[idx + 1]
    if task_name is None:
        task_name = "ShadowHandOver"
        sys.argv.extend(["--task", task_name])

    # Validate: cfg YAML must exist
    cfg_yaml = os.path.join(BIDEXHANDS_DIR, "cfg", f"{task_name}.yaml")
    if not os.path.exists(cfg_yaml):
        print(f"[Error] Unknown task: {task_name}")
        print(f"  No config found: {cfg_yaml}")
        sys.exit(1)

    # --- Ensure headless ---
    if "--headless" not in sys.argv:
        sys.argv.append("--headless")

    # --- Banner (partial — num_agents determined after env creation) ---
    print("=" * 60)
    print("  Bi-DexHands Unified ZMQ Server")
    print("=" * 60)
    print(f"  Task      : {task_name}")
    print(f"  Host      : {SERVER_HOST}")
    print(f"  Port      : {SERVER_PORT}")
    print("=" * 60)

    server_socket = None
    context = None

    try:
        # --- ZMQ setup -----------------------------------------------------
        context = zmq.Context()
        server_socket = context.socket(zmq.REP)
        server_socket.bind(f"tcp://{SERVER_HOST}:{SERVER_PORT}")
        print(f"[Server] ZMQ bound to tcp://{SERVER_HOST}:{SERVER_PORT}")

        # --- IsaacGym environment ------------------------------------------
        args = get_args()
        args.num_envs = 1
        args.compute_device_id = 0
        args.graphics_device_id = 0

        cfg, cfg_train, logdir = load_cfg(args)
        cfg["env"]["numEnvs"] = 1

        # Asset root → absolute path
        assets_root = os.path.join(SCRIPT_DIR, "assets")
        if os.path.exists(assets_root):
            cfg["env"]["asset"]["assetRoot"] = assets_root

        sim_params = parse_sim_params(args, cfg, cfg_train)

        # Agent index: MultiAgent needs get_AgentIndex, SingleAgent uses 0
        if args.task_type == "MultiAgent":
            agent_index = get_AgentIndex(cfg)
        else:
            agent_index = 0

        task_obj, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
        device = env.rl_device if hasattr(env, "rl_device") else torch.device("cuda:0")

        adapter = EnvAdapter(env, task_obj, args.task_type, device)

        # Initial reset to determine dimensions
        init_obs = adapter.reset()
        obs_dim = int(init_obs.shape[0])
        act_dim = env.num_agents * env.num_acts if env.num_agents > 1 else env.num_acts
        episode_length = cfg["env"]["episodeLength"]

        print(f"[Server] Task Type: {args.task_type} ({env.num_agents} agent(s))")
        print(f"[Server] Device: {device}")
        print(f"[Server] Obs dim: {obs_dim}, Act dim: {act_dim}")
        print(f"[Server] Episode length: {episode_length}")
        print(f"[Server] Ready! Waiting for client...")

        # --- Main loop ------------------------------------------------------
        step_count = 0

        while True:
            try:
                message = server_socket.recv()
                cmd, data = compat_loads(message)
                response = None

                if cmd == "reset":
                    print("[Server] reset")
                    obs = adapter.reset()
                    response = ("ok", obs)
                    print(f"[Server] reset done | obs shape: {obs.shape}")

                elif cmd == "step":
                    step_count += 1
                    obs, rew, terminated, truncated, info = adapter.step(data)
                    response = ("ok", (obs, rew, terminated, truncated, info))
                    if step_count % 50 == 0:
                        print(
                            f"[Server] step {step_count} | "
                            f"rew: {rew:.4f} | mass: {info.get('mass', 'N/A')}"
                        )

                elif cmd == "info":
                    response = ("ok", {
                        "task": task_name,
                        "task_type": args.task_type,
                        "num_agents": env.num_agents,
                        "obs_dim": obs_dim,
                        "act_dim": act_dim,
                        "episode_length": episode_length,
                    })

                elif cmd == "close":
                    print("[Server] Shutting down (client requested)...")
                    response = ("ok", None)
                    server_socket.send(pickle.dumps(response))
                    break

                else:
                    response = ("error", f"Unknown command: {cmd}")

                server_socket.send(pickle.dumps(response))

            except KeyboardInterrupt:
                print("\n[Server] Interrupted")
                break
            except Exception as e:
                print(f"[Server] Error: {e}")
                traceback.print_exc()
                try:
                    server_socket.send(pickle.dumps(("error", str(e))))
                except:
                    pass

    except Exception as e:
        print(f"[Server] Fatal: {e}")
        traceback.print_exc()
    finally:
        if server_socket:
            server_socket.close()
        if context:
            context.term()
        print("[Server] Shutdown complete")


if __name__ == "__main__":
    run_server()
