#!/bin/bash

cd ${CATKIN_WS}/
rosrun joy joy_node _dev:=/dev/input/js0 _autorepeat_rate:=30 _joy_type:="xbox_wired"
