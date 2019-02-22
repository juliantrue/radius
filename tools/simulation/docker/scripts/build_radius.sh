#!/bin/bash

RADIUS_PATH=$1
if [[ -z "${RADIUS_PATH}" ]]; then
    echo "First argument is missing."
    echo "Usage  : build_radius.sh <full_path_to_radius>"
    echo "Example: build_radius.sh /data/src/radius"
    exit 1
fi

cd ${CATKIN_WS}

if [[ ! -L "${CATKIN_WS}/src/mrcnn" ]]; then
    cp -r ${RADIUS_PATH}/ros/packages/mrcnn ${CATKIN_WS}/src/
fi

if [[ ! -L "${CATKIN_WS}/src/RADIUS_debug" ]]; then
    cp -r ${RADIUS_PATH}/ros/packages/radius_debug ${CATKIN_WS}/src/
fi

if [[ ! -L "${CATKIN_WS}/src/lndng_controller" ]]; then
    cp -r ${RADIUS_PATH}/ros/packages/lndng_controller ${CATKIN_WS}/src/
fi

catkin_make

echo "Marking as executable"
chmod +x ${CATKIN_WS}/src/mrcnn/src/mrcnn_node.py
chmod +x ${CATKIN_WS}/src/lndng_controller/src/lndng_controller_node.py
chmod +x ${CATKIN_WS}/src/radius_debug/src/radius_debug.py
