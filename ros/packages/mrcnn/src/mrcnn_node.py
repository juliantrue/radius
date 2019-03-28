#!/usr/bin/python
from __future__ import division
from __future__ import print_function
# Ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Python libs
import sys, os, time
from multiprocessing import Lock
import numpy as np
import cv2

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
    DETECTION_MIN_CONFIDENCE = 0.9950
    """
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    """
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
	
	def preprocess(self, img):
		




	
		return img



	def compute_masks(self, img):
		rospy.loginfo("Computing detections.")
		with self._mutex:
			start = time.time()
			results = self.detect([img], verbose=0)
			end = time.time()
			rospy.loginfo("Inferences computed in: {:.2f}ms".format((end-start)*10**3))
			r = results[0]
			# Extract specified class
			if(self._class_names.index(self.only_class) in r['class_ids']):
				idx = np.where(r['class_ids'] == self._class_names.index(self.only_class))[0]
				print(r['scores'][idx])
				mask, masked_img = display_instances(img,
                                                     np.array(r['rois'][idx,:]),
                                                     np.array(r['masks'][:,:,idx]),
                                                     np.array(r['class_ids'][idx]),
                                                     self._class_names,
                                                     np.array(r['scores'][idx]))
			else:
				rospy.logwarn('No instances detected. Passing raw image, no mask.')
				masked_img = img
				mask = np.zeros(img.shape, dtype=np.uint8)
				mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

		return mask, masked_img

class Data_Handler(object):
    def __init__(self):
        self.data = None
        self.data_available = False
        self._bridge = CvBridge()
        self._mask_topic_pub = None
        self._img_mask_pub = None

    def on_img(self, img_msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if not self.data_available:
            rospy.loginfo("Data available!")
            self.data = cv_image
            self.data_available = True

    def publish_mask(self, mask):
        mask_msg = self._bridge.cv2_to_imgmsg(mask, "mono8")
        self._mask_topic_pub.publish(mask_msg)

    def publish_img_mask(self, img_mask):
        img_mask_msg = self._bridge.cv2_to_imgmsg(img_mask, "bgr8")
        self._img_mask_topic_pub.publish(img_mask_msg)

    def set_mask_pub(self, pub):
        self._mask_topic_pub = pub

    def set_img_mask_pub(self, pub):
        self._img_mask_topic_pub = pub

def main():
    #TODO: Elevate these to arguments
	root_dir = os.getcwd()
    #model_path = os.path.join(root_dir, "src/mrcnn/src/model/mask_rcnn_coco.h5" 
	model_path = os.path.join(root_dir, "src/mrcnn/src/model/mask_rcnn_radius.h5")


    # Setup config for inferencing
	RedtailConfig = InferenceConfig()
	RedtailConfig.display()

    # Create data handler object to prevent storing as internal state
	dh = Data_Handler()

    # Initialize the model to inference on COCO data
    # Initialize model and load weights prior to spinning up node
    # Only look for one class to avoid issues with controller node
	model = Mask_RCNN(mode='inference', config=RedtailConfig,
					  model_dir='./model', class_name='elephant')
	print("Loading weights {}".format(model_path))
	model.load_weights(model_path, by_name=True)


	rospy.init_node('Mask_RCNN', log_level=rospy.DEBUG)
	rospy.loginfo("Starting Mask_RCNN node...")

	# Subscribe to camera input topic
	rospy.Subscriber("camera/image_raw", Image, dh.on_img)
	rospy.loginfo("Listening on camera/image_raw")

	# Publish to relevent topics
	dh.set_mask_pub(rospy.Publisher('radius/mask_rcnn/mask', Image, queue_size=20))
	dh.set_img_mask_pub(rospy.Publisher('radius/mask_rcnn/img_mask', Image, queue_size=5))

	rate = rospy.Rate(2)
	while not rospy.is_shutdown():
		if(dh.data_available):
			rospy.loginfo("Preparing for processing")
			img = model.preprocess(dh.data)
			mask, masked_img = model.compute_masks(img)
			dh.publish_img_mask(masked_img)
			dh.publish_mask(mask)
			dh.data_available = False
		rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
