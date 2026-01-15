# Dockerfile

# 1. Base Image: NVIDIA CUDA + Ubuntu 20.04
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# 2. Environment Variables
# (Set basic envs only. Specific envs like VK_ICD_FILENAMES are handled in setup.sh)
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV DISPLAY=:1

# 3. Install Essential Packages
# (Includes Python, Git, VNC server, XFCE desktop, OpenGL libs, etc.)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    vim \
    wget \
    xfce4 \
    xfce4-goodies \
    tigervnc-standalone-server \
    tigervnc-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Python alias (python -> python3)
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --no-cache-dir wandb

# 5. Run System Setup Script
# (Copies setup.sh to container and executes it to configure VNC/Terminal/.bashrc)
WORKDIR /root
COPY setup.sh /root/
RUN chmod +x /root/setup.sh && /root/setup.sh

# 6. Create VNC Entrypoint Script
# (Handles lock file cleanup and starts the VNC server)
RUN echo '#!/bin/bash\n\
# Remove existing lock files to prevent VNC startup errors\n\
rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1 /root/.vnc/*.log /root/.vnc/*.pid\n\
\n\
# Start VNC Server (No localhost restriction)\n\
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no\n\
\n\
# Keep container running\n\
tail -f /dev/null' > /usr/local/bin/start_vnc.sh && \
chmod +x /usr/local/bin/start_vnc.sh

# 7. Set Final Working Directory
# (Sets the default directory to /data for easy access to mounted volumes)
WORKDIR /data

# 8. Container Startup Command
CMD ["/usr/local/bin/start_vnc.sh"]
