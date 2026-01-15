# DexterousHands Environment Server (Dockerized)

This repository serves as a **Dockerized environment server** for training and testing dexterous manipulation tasks, modified from [PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands).

It is designed to run with **NVIDIA Isaac Gym** (preview 4) and includes a custom server script (`server_env.py`) for external communication.

## 🚀 Setup & Installation Guide

### 1. Prepare Isaac Gym
NVIDIA Isaac Gym cannot be distributed via Docker Hub due to licensing. You must download and extract it manually.

1. Download **`IsaacGym_Preview_4_Package.tar.gz`** from the [NVIDIA Developer Website](https://developer.nvidia.com/isaac-gym).
2. Place the file in your workspace directory and extract it.

```bash
# Example: Setup in 'my_workspace'
mkdir my_workspace && cd my_workspace

# Place the downloaded .tar.gz here, then run:
tar -xvzf IsaacGym_Preview_4_Package.tar.gz

# ⚠️ Ensure 'isaacgym' folder exists in your current path
ls
# Output should be: isaacgym/  IsaacGym_Preview_4_Package.tar.gz

```

### 2. Run Docker Container

Execute the following command in the directory where `isaacgym` is located. We mount the current directory to `/data` so the container can access Isaac Gym and your code.

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

* **`-v $(pwd):/data`**: Maps the host current directory to `/data` in the container.
* **`-p 5901:5901`**: VNC Port (Connect via `localhost:5901`, password: `password`).

### 3. Install Python Dependencies (Inside Container)

Since the code and Isaac Gym are mounted from the host, you need to manually install them **once** inside the container.

```bash
# 1. Enter the container
docker exec -it bidexhands /bin/bash

# 2. Install Isaac Gym (Essential)
cd /data/isaacgym/python
pip install -e .

# 3. Install DexterousHands Environment
# (Assuming you cloned this repo into 'DexterousHands-Env-Server' folder)
cd /data/DexterousHands-Env-Server
pip install -e .

```

### 4. Run the Server

After installation, you can run the server script.

```bash
# Inside the container:
cd /data/DexterousHands-Env-Server
python server_env.py

```
