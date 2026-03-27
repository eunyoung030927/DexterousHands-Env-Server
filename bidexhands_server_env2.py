"""
Unified ZMQ Server for Bi-DexHands IsaacGym Environments (v2 — parallel eval)

Supports both SingleAgent and MultiAgent tasks via --task flag.
Supports parallel evaluation: client sends ("init", {"num_envs": N}) to create
N environments on GPU.  If no "init" is received before the first "reset",
the server falls back to num_envs=1 (legacy single-env mode).

Usage:
    python bidexhands_server_env2.py --task ShadowHandCatchOver2Underarm
    python bidexhands_server_env2.py --task ShadowHandOver
    python bidexhands_server_env2.py --task ShadowHandBottleCap --task_type MultiAgent
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
    """Convert info dict values from tensors to serializable types (single env)."""
    info = {}
    if isinstance(info_raw, dict):
        for k, v in info_raw.items():
            if isinstance(v, torch.Tensor):
                info[k] = v.cpu().detach().numpy().tolist()
            else:
                info[k] = v
    return info


def _convert_info_batch(info_raw, num_envs):
    """Convert info dict to a list of per-env dicts for batch mode."""
    infos = [{} for _ in range(num_envs)]
    if isinstance(info_raw, dict):
        for k, v in info_raw.items():
            if isinstance(v, torch.Tensor):
                arr = v.cpu().detach().numpy()
                for i in range(num_envs):
                    if arr.ndim >= 1 and arr.shape[0] >= num_envs:
                        infos[i][k] = arr[i].tolist()
                    else:
                        infos[i][k] = arr.tolist()
            else:
                for i in range(num_envs):
                    infos[i][k] = v
    return infos


# ---------------------------------------------------------------------------
# EnvAdapter — absorbs SingleAgent / MultiAgent differences
# ---------------------------------------------------------------------------
class EnvAdapter:
    """Unified interface over VecTaskPython (single) and MultiVecTaskPython (multi).

    num_envs == 1 (legacy):
        - reset()  → 1-D numpy obs (obs_dim,)
        - step(action_flat) → (obs, reward, terminated, truncated, info)

    num_envs > 1 (batch):
        - reset()  → 2-D numpy obs (num_envs, obs_dim)
        - step(actions) → (obs, rewards, terminated, truncated, infos)
    """

    def __init__(self, env, task, task_type, device, num_envs):
        self.env = env
        self.task = task
        self.is_multi_agent = (task_type == "MultiAgent")
        self.num_agents = env.num_agents
        self.action_dim_per_agent = env.num_acts
        self.device = device
        self.num_envs = num_envs

    # ----- reset ----------------------------------------------------------
    def reset(self) -> np.ndarray:
        if self.is_multi_agent:
            return self._reset_multi()
        return self._reset_single()

    def _reset_single(self):
        # VecTaskPython.reset() → tensor (num_envs, obs_dim)
        obs = _to_numpy(self.env.reset())
        if self.num_envs == 1:
            if obs.ndim == 2:
                obs = obs[0]  # (obs_dim,)
        # num_envs > 1: obs already (num_envs, obs_dim)
        return obs

    def _reset_multi(self):
        # MultiVecTaskPython.reset() → (obs, state_all, None)
        # obs shape: (num_envs, num_agents, obs_dim)
        result = self.env.reset()
        obs = _to_numpy(result[0])
        if self.num_envs == 1:
            obs = obs.flatten()  # (num_agents * obs_dim,)
        else:
            # (num_envs, num_agents, obs_dim) → (num_envs, num_agents * obs_dim)
            obs = obs.reshape(self.num_envs, -1)
        return obs

    # ----- step -----------------------------------------------------------
    def step(self, action_data):
        if self.num_envs == 1:
            if self.is_multi_agent:
                return self._step_multi_single(action_data)
            return self._step_single_single(action_data)
        else:
            if self.is_multi_agent:
                return self._step_multi_batch(action_data)
            return self._step_single_batch(action_data)

    # --- num_envs == 1: legacy single-env interface -----------------------
    def _step_single_single(self, action_flat):
        action_tensor = torch.tensor(
            np.array(action_flat, dtype=np.float32),
            dtype=torch.float, device=self.device,
        ).unsqueeze(0)  # (act_dim,) → (1, act_dim)

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

    def _step_multi_single(self, action_flat):
        arr = np.array(action_flat, dtype=np.float32).flatten()
        dim = self.action_dim_per_agent

        actions_list = []
        for i in range(self.num_agents):
            a = torch.tensor(
                arr[i * dim : (i + 1) * dim],
                dtype=torch.float, device=self.device,
            ).unsqueeze(0)
            actions_list.append(a)

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

    # --- num_envs > 1: batch interface ------------------------------------
    def _step_single_batch(self, actions):
        # actions: (num_envs, act_dim)
        action_tensor = torch.tensor(
            np.array(actions, dtype=np.float32),
            dtype=torch.float, device=self.device,
        )  # (num_envs, act_dim)

        obs_t, rew_t, done_t, info_raw = self.env.step(action_tensor)

        obs = _to_numpy(obs_t)                                  # (num_envs, obs_dim)
        rewards = _to_numpy(rew_t).flatten()                     # (num_envs,)
        terminated = _to_numpy(done_t).flatten().astype(bool)    # (num_envs,)
        truncated = np.zeros(self.num_envs, dtype=bool)

        infos = _convert_info_batch(info_raw, self.num_envs)
        return obs, rewards, terminated, truncated, infos

    def _step_multi_batch(self, actions):
        # actions: (num_envs, num_agents * act_dim_per_agent)
        arr = np.array(actions, dtype=np.float32)
        dim = self.action_dim_per_agent

        # Split into per-agent tensors: each (num_envs, act_dim_per_agent)
        actions_list = []
        for i in range(self.num_agents):
            a = torch.tensor(
                arr[:, i * dim : (i + 1) * dim],
                dtype=torch.float, device=self.device,
            )
            actions_list.append(a)

        obs_t, _state, rew_t, done_t, info_raw, _ = self.env.step(actions_list)

        # obs: (num_envs, num_agents, obs_dim) → (num_envs, num_agents * obs_dim)
        obs = _to_numpy(obs_t).reshape(self.num_envs, -1)

        # rewards: mean across agents per env → (num_envs,)
        rew_np = _to_numpy(rew_t)
        if rew_np.ndim >= 2:
            rewards = rew_np.mean(axis=-1).flatten()
        else:
            rewards = rew_np.flatten()

        # terminated: any agent done per env → (num_envs,)
        done_np = _to_numpy(done_t)
        if done_np.ndim >= 2:
            terminated = done_np.any(axis=-1).flatten().astype(bool)
        else:
            terminated = done_np.flatten().astype(bool)

        truncated = np.zeros(self.num_envs, dtype=bool)

        infos = _convert_info_batch(info_raw, self.num_envs)
        return obs, rewards, terminated, truncated, infos


# ---------------------------------------------------------------------------
# Environment factory — creates Isaac Gym env with given num_envs
# ---------------------------------------------------------------------------
def _create_env(num_envs):
    """Create Isaac Gym environment and return (adapter, metadata)."""
    args = get_args()
    args.num_envs = num_envs
    args.compute_device_id = 0
    args.graphics_device_id = 0

    cfg, cfg_train, logdir = load_cfg(args)
    cfg["env"]["numEnvs"] = num_envs

    # Asset root → absolute path
    assets_root = os.path.join(SCRIPT_DIR, "assets")
    if os.path.exists(assets_root):
        cfg["env"]["asset"]["assetRoot"] = assets_root

    sim_params = parse_sim_params(args, cfg, cfg_train)

    if args.task_type == "MultiAgent":
        agent_index = get_AgentIndex(cfg)
    else:
        agent_index = 0

    task_obj, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)
    device = env.rl_device if hasattr(env, "rl_device") else torch.device("cuda:0")

    adapter = EnvAdapter(env, task_obj, args.task_type, device, num_envs)

    # Initial reset to determine dimensions
    init_obs = adapter.reset()
    if num_envs == 1:
        obs_dim = int(init_obs.shape[0])
    else:
        obs_dim = int(init_obs.shape[1])
    act_dim = env.num_agents * env.num_acts if env.num_agents > 1 else env.num_acts
    episode_length = cfg["env"]["episodeLength"]

    metadata = {
        "task": args.task,
        "task_type": args.task_type,
        "num_envs": num_envs,
        "num_agents": env.num_agents,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "episode_length": episode_length,
    }

    print(f"[Server] Environment created: num_envs={num_envs}")
    print(f"[Server] Task Type: {args.task_type} ({env.num_agents} agent(s))")
    print(f"[Server] Device: {device}")
    print(f"[Server] Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"[Server] Episode length: {episode_length}")

    return adapter, metadata


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

    # --- Banner ---
    print("=" * 60)
    print("  Bi-DexHands Unified ZMQ Server (v2 — parallel eval)")
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
        print(f"[Server] Waiting for client (env not created yet)...")

        # --- Lazy init: env created on "init" or first "reset" -------------
        adapter = None
        metadata = None
        step_count = 0

        # --- Main loop ------------------------------------------------------
        while True:
            try:
                message = server_socket.recv()
                cmd, data = compat_loads(message)
                response = None

                if cmd == "init":
                    num_envs = 1
                    if isinstance(data, dict):
                        num_envs = data.get("num_envs", 1)
                    if adapter is not None and adapter.num_envs == num_envs:
                        print(f"[Server] Re-init (same num_envs={num_envs}), reusing env...")
                        step_count = 0
                        adapter.reset()
                        response = ("ok", metadata)
                    elif adapter is not None:
                        response = ("error",
                                    f"Cannot change num_envs from {adapter.num_envs} to {num_envs} "
                                    f"without server restart (PhysX limitation).")
                    else:
                        print(f"[Server] init requested: num_envs={num_envs}")
                        adapter, metadata = _create_env(num_envs)
                        response = ("ok", metadata)
                    print(f"[Server] Ready!")

                elif cmd == "reset":
                    if adapter is None:
                        print("[Server] No init received, creating env with num_envs=1 (legacy)")
                        adapter, metadata = _create_env(1)
                    print("[Server] reset")
                    obs = adapter.reset()
                    response = ("ok", obs)
                    print(f"[Server] reset done | obs shape: {obs.shape}")

                elif cmd == "step":
                    if adapter is None:
                        response = ("error",
                                    "Environment not initialized. "
                                    "Send 'init' or 'reset' first.")
                    else:
                        step_count += 1
                        result = adapter.step(data)
                        if adapter.num_envs == 1:
                            obs, rew, terminated, truncated, info = result
                            response = ("ok", (obs, rew, terminated, truncated, info))
                            if step_count % 50 == 0:
                                print(
                                    f"[Server] step {step_count} | "
                                    f"rew: {rew:.4f} | mass: {info.get('mass', 'N/A')}"
                                )
                        else:
                            obs, rewards, terminated, truncated, infos = result
                            response = ("ok", (obs, rewards, terminated, truncated, infos))
                            if step_count % 50 == 0:
                                print(
                                    f"[Server] step {step_count} | "
                                    f"mean_rew: {rewards.mean():.4f}"
                                )

                elif cmd == "info":
                    if metadata is not None:
                        response = ("ok", metadata)
                    else:
                        response = ("ok", {
                            "task": task_name,
                            "num_envs": 0,
                            "status": "not_initialized",
                        })

                elif cmd == "render":
                    response = ("ok", np.array([]))

                elif cmd == "close":
                    print("[Server] Client disconnected, resetting step count...")
                    step_count = 0
                    response = ("ok", None)

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
