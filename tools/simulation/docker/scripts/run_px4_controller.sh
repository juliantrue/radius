#!/bin/bash

cd ${CATKIN_WS}/
rosrun px4_controller px4_controller_node _altitude_gain:=2
