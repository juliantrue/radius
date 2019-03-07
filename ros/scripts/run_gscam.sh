#!/bin/bash

cd ${CATKIN_WS}
export GSCAM_CONFIG="v4l2src device=/dev/video1 ! video/x-raw, width=640, height=360 ! ffmpegcolor space"
rosrun gscam gscam
