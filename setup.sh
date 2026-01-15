#!/bin/bash
# setup.sh

# --------------------------------------------------------
# 1. Install additional terminal utilities & Configure default terminal
# --------------------------------------------------------
apt-get update && apt-get install -y xfce4-terminal dbus-x11 xterm

# Set xfce4-terminal as the default terminal emulator
update-alternatives --set x-terminal-emulator /usr/bin/xfce4-terminal.wrapper

# --------------------------------------------------------
# 2. VNC Setup (Password & Startup script)
# --------------------------------------------------------
# Create .vnc directory
mkdir -p /root/.vnc

# Set VNC password to 'password' automatically
echo "password" | vncpasswd -f > /root/.vnc/passwd
chmod 600 /root/.vnc/passwd

# Create xstartup file to launch XFCE4 session
cat <<EOF > /root/.vnc/xstartup
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
/usr/bin/startxfce4
EOF

# Make xstartup executable
chmod +x /root/.vnc/xstartup

# --------------------------------------------------------
# 3. Isaac Gym Environment Variable (Runtime configuration)
# --------------------------------------------------------
# Append the export command to .bashrc so it applies when the user logs in.
# This avoids build-time errors when the file doesn't exist yet.
echo 'export VK_ICD_FILENAMES=/data/isaacgym/docker/nvidia_icd.json' >> /root/.bashrc
