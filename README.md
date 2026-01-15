# DexterousHands Environment Server

This repository serves as a **Dockerized environment server** for training and testing dexterous manipulation tasks, modified from [PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands).

It is designed to run with **NVIDIA Isaac Gym** and includes a custom server script (`server_env.py`) for external communication.

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
* **`-p 5901:5901`**: VNC Port for visual monitoring.
* **`-p 5000:5000`**: Server communication port.

### 3. Setup Inside Container (Important)

You need to set your VNC password and install the dependencies manually once inside the container.

```bash
# 1. Enter the container
docker exec -it bidexhands /bin/bash

# 2. Set VNC Password (REQUIRED)
# Run this command and set your own password for remote connection
vncpasswd

# 3. Install Isaac Gym
cd /data/isaacgym/python
pip install -e .

# 4. Install DexterousHands Environment
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

### 5. Connect via VNC (Visual Monitor)

You can view the simulation GUI remotely.

1. Open your **VNC Viewer**.
2. Connect to: `localhost:5901`
3. Password: **The password you set in Step 3.**





## ℹ️ Troubleshooting

### 1. Client-Server Connection (Network Setup)
If the external client (e.g., `sb3-diffusion`) cannot connect to the server, you need to manually configure the IP address.

**Step 1: Get the Server Container IP**
Run this command on your host machine to find the IP of the running container:
```bash
docker inspect bidexhands | grep IPAddress
# Output Example: "IPAddress": "172.17.0.2"

```

**Step 2: Update Client Code**
Open the file `sb3-diffusion/sb3_diffusion/common/env_utils.py` in your client repository.
Find the `case "bidexhands":` block and update `SERVER_IP` with the address found above.

```python
# sb3_diffusion/common/env_utils.py

case "bidexhands":
    # ⚠️ Update this with your container's actual IP
    SERVER_IP = "172.17.0.x"  
    PORT = 5000

```
