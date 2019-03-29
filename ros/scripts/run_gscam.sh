#!/bin/bash

cd ${CATKIN_WS}
export GSCAM_CONFIG="v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! ffmpegcolor space"
rosrun gscam gscam
