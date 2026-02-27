#!/bin/bash

# Kill any ghost processes
sudo pkill -9 rpicam-vid
sudo pkill -9 ffmpeg

# 1. Enable the virtual camera module
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=10 card_label="ShrimpSenseCam" exclusive_caps=0

# 2. Run the camera command with a more compatible format for OpenCV
rpicam-vid -t 0 --width 1280 --height 720 --framerate 30 --codec yuv420 --inline -n -o - | \
ffmpeg -f rawvideo -vcodec rawvideo -pixel_format yuv420p -video_size 1280x720 -framerate 30 -i - \
-f v4l2 -pix_fmt yuyv422 /dev/video10 &

# 3. Wait for stabilization
sleep 3

# 4. Environment Variables
export QT_QPA_PLATFORM=wayland
cd /home/hiponpd/Documents/GitHub/ShrimpMachineApp
/home/hiponpd/Documents/GitHub/ShrimpMachineApp/venv/bin/python3 app.py

# 5. Cleanup
pkill -f rpicam-vid
pkill -f ffmpeg