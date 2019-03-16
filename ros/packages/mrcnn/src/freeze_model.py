from __future__ import print_function
from __future__ import division
import sys, os, time
from multiprocessing import Lock
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Mask RCNN stuff
sys.path.append('./mrcnn')
from mrcnn.config import Config
from mrcnn.visualize import display_instances
from mrcnn import model as modellib


class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    #NUM_CLASSES = 1 + 80 # COCO has 80 classes
    NUM_CLASSES = 1+1

class InferenceConfig(CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    class_names = ['BG', 'elephant']

class Mask_RCNN(modellib.MaskRCNN):
    # Thread safe ros implementation of the Mask RCNN architecture
    # implementation by matterport in Keras
    # Finds only one class as per our application
    def __init__(self, mode, config, model_dir, class_name):
        self.only_class = class_name
        self._mutex = Lock()
        self._class_names = config.class_names
        super(Mask_RCNN, self).__init__(mode, config, model_dir)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


root_dir = os.getcwd()
model_path = os.path.join(root_dir, "model/mask_rcnn_radius.h5")

# Setup config for inferencing
RedtailConfig = InferenceConfig()
RedtailConfig.display()

print("Creating Model")
model = Mask_RCNN(mode='inference', config=RedtailConfig,
                  model_dir='./model', class_name='elephant')
print("Loading weights")
model.load_weights(model_path, by_name=True)

# Freeze graph
print("Freezing graph...")
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.keras_model.outputs])
print("Graph frozen")

# Write to protocol buffer
print("Writing frozen graph to file")
pb_model_dir = os.path.join(root_dir, "model")
tf.train.write_graph(frozen_graph, pb_model_dir, "mask_rcnn_radius.pb", as_text=False)



