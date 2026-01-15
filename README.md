# DexterousHands Environment Server (Dockerized)

This repository serves as a **Dockerized environment server** for training and testing dexterous manipulation tasks. It is modified based on the original [PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands) repository.

We have added a custom **server script (`server_env.py`)** to communicate with external clients (e.g., `sb3-diffusion` models) and configured a Docker environment with **VNC support** for remote visual monitoring.

## 🛠 Prerequisites

Before running the container, ensure you have the following installed on your host machine:

* **NVIDIA Driver**
* **Docker** & **NVIDIA Container Toolkit**
* **NVIDIA Isaac Gym** (Must be downloaded manually due to licensing)

---

## 📦 Installation & Setup Guide

### 1. Download NVIDIA Isaac Gym
Since Isaac Gym cannot be distributed via Docker Hub, you must download it manually and place it in your workspace.

1. Download **`IsaacGym_Preview_4_Package.tar.gz`** from the [NVIDIA Developer Website](https://developer.nvidia.com/isaac-gym).
2. Create a workspace directory and extract the package there.

```bash
# Example setup
mkdir my_workspace && cd my_workspace

# Place the .tar.gz file here and extract it
tar -xvzf IsaacGym_Preview_4_Package.tar.gz

# Verify that the 'isaacgym' folder exists
ls
# Output should look like: isaacgym/  IsaacGym_Preview_4_Package.tar.gz

```

### 2. Clone This Repository

Clone this repository into the **same workspace directory** where `isaacgym` is located.

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[YOUR_GITHUB_ID]/DexterousHands-Env-Server.git

# (Optional) Rename the directory for convenience (e.g., bidexhands)
# mv DexterousHands-Env-Server bidexhands

```

### 3. Check Directory Structure

Your workspace should look like this before running Docker:

```text
my_workspace/
├── isaacgym/                   # From NVIDIA (Essential)
├── DexterousHands-Env-Server/  # This Repository (Source Code)
└── IsaacGym_Preview_4_Package.tar.gz

```

---

## 🚀 How to Run

### 1. Start the Docker Container

Run the following command inside your workspace directory (`my_workspace`).

We mount the current directory (`$(pwd)`) to `/data` inside the container. This ensures that both the **Isaac Gym** environment and your **source code** are accessible to the container.

```bash
docker run -d \
  --gpus all \
  --name bidexhands \
  --shm-size=16g \
  -p 5901:5901 \
  -p 5000:5000 \
  -v $(pwd):/data \
  eunyoung927/bidexhands-issacgym:v1

```

**Flag Explanations:**

* `eunyoung927/bidexhands-issacgym:v1`: The pre-built Docker image with dependencies installed.
* `-v $(pwd):/data`: Maps your local workspace to `/data` in the container.
* `-p 5901:5901`: Exposes the VNC port for visual monitoring.
* `-p 5000:5000`: Exposes the port for `server_env.py` communication.
* `--shm-size=16g`: Prevents memory crashes during simulation.

### 2. Connect via VNC (Visual Monitor)

You can view the simulation GUI remotely.

1. Open your **VNC Viewer**.
2. Connect to: `localhost:5901`
3. Password: `password`

### 3. Run the Server Code

Enter the container and execute the server script.

```bash
# 1. Enter the container
docker exec -it bidexhands /bin/bash

# 2. Navigate to the repository folder
# (Replace 'DexterousHands-Env-Server' with your actual cloned folder name)
cd /data/DexterousHands-Env-Server

# 3. Run the server
python server_env.py

```

---

## ℹ️ Troubleshooting

* **`python: can't open file ... No such file or directory`**:
* This happens if the volume mount is incorrect. Make sure you run the `docker run` command from the **parent directory** that contains both `isaacgym` and the repository folder.


* **Simulator Crash (Vulkan/ICD Error)**:
* Ensure `isaacgym` is correctly located at `/data/isaacgym` inside the container. The environment expects the file `/data/isaacgym/docker/nvidia_icd.json` to exist.



## 📂 Reference

* Original Repository: [PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands)
* Isaac Gym: [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)

```
