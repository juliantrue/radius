#!/bin/bash

cd ${CATKIN_WS}/
MODEL_PATH="/data/redtail/models/pretrained/yolo-relu.caffemodel"
PROTOTXT_PATH="/data/redtail/models/pretrained/yolo-relu.prototxt"

if[ -f ${MODEL_PATH} ]; then
  rosrun caffe_ros caffe_ros_node __name:=object_dnn _prototxt_path:=$PROTOTXT_PATH _model_path:=${MODEL_PATH} _output_layer:=fc25 _inp_scale:=0.00390625 _inp_fmt:="RGB" _post_proc:="YOLO" _obj_det_threshold:=0.2 _use_fp16:=true
else
  cat /data/redtail/models/pretrained/yolo-relu.caffemodel.* > /data/redtail/models/pretrained/yolo-relu.caffemodel
  rosrun caffe_ros caffe_ros_node __name:=object_dnn _prototxt_path:=$PROTOTXT_PATH _model_path:=${MODEL_PATH} _output_layer:=fc25 _inp_scale:=0.00390625 _inp_fmt:="RGB" _post_proc:="YOLO" _obj_det_threshold:=0.2 _use_fp16:=true
fi
