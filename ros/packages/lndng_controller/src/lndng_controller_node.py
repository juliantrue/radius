#!/usr/bin/python
from __future__ import division
from __future__ import print_function
# Ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Python libs
import sys, os, datetime, time, struct, math
import yaml
from multiprocessing import Lock
import numpy as np
import cv2

# Parse config file
config_path = os.path.join(os.getcwd(), "src/lndng_controller/src/config.yml")
with open(config_path, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

LOGDIR_PATH= os.path.join(cfg["LOGDIR"], "{}".format(datetime.datetime.now()))

import functools
def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
	start_time = time.time()
	value = func(*args,**kwargs)
	end_time = time.time()
	run_time = (end_time - start_time)*10**3
	rospy.logdebug("{} runtime: {:.4f}ms".format(func.__name__, run_time))
	return value
    return wrapper_timer

def compose_debug_img(debug_img, curr_delta):
    size = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    h,w,_ = debug_img.shape
    center_pixel = (int(math.floor(debug_img.shape[1]/2.0)) + 
			int(math.floor(debug_img.shape[1]/4.0)),
                    int(math.floor(debug_img.shape[0]/2.0)))

    # Overlay errors
    errors_colour = (0,0,255)
    cv2.putText(debug_img, "Delta_x: {}".format(curr_delta.dx), (10,h-10), 
		font, size,errors_colour,1,cv2.LINE_AA) 
    cv2.putText(debug_img, "Delta_y: {}".format(curr_delta.dy), (10,h-30), 
		font, size,errors_colour,1,cv2.LINE_AA)
    cv2.putText(debug_img, "Delta_theta: {}".format(curr_delta.dtheta*180/math.pi), (10,h-50), 
		font, size,errors_colour,1,cv2.LINE_AA)

    # Overlay manhattan distance error lines
    cv2.line(debug_img, center_pixel, (center_pixel[0]+curr_delta.dx,
	     center_pixel[1]),(0,0,255),1)
    cv2.line(debug_img, (center_pixel[0]+curr_delta.dx,
			 center_pixel[1]), (center_pixel[0]+curr_delta.dx,
	     center_pixel[1]+curr_delta.dy),(0,0,255),1)

    return debug_img

class Delta(object):
    def __init__(self, delta_x, delta_y, delta_theta, delta_theta_raw, is_mask_referenced):
	# Store deltas
	self.dx = delta_x
	self.dy = delta_y
	self.dtheta_raw = delta_theta_raw
	self.dtheta = delta_theta
	self.errors = (self.dx, self.dy, self.dtheta)

	# Delta metadata
	self.is_mask_referenced = is_mask_referenced
	
class Lander(object):
    def __init__(self, control_img):
        self._control_img = control_img
	self._last_delta_buffer = []
        self._delta_buffer = [] 
	self._alpha = cfg["CONTROLLER"]["ALPHA"]
        self._DELTA_BUFFER_MAX_SIZE = cfg["CONTROLLER"]["ERROR_HISTORY_BUFFER_SIZE"]
	self._window_length = cfg["CONTROLLER"]["WINDOW_LENGTH"]
	self._MAX_NO_LOCK_CYCLES = cfg["CONTROLLER"]["MAX_NO_LOCK_CYCLES"]
	self.DNN_no_lock_counter = 0
	self.DNN_locked = False

    def reset_lock_counter(self):
	self.DNN_locked = True
	rospy.loginfo("DNN locked.")
	self.DNN_no_lock_counter = 0

    def increment_no_lock_counter(self):
	self.DNN_no_lock_counter += 1
	if self.DNN_no_lock_counter >= self._MAX_NO_LOCK_CYCLES:
	    rospy.logwarn("DNN lock lost")
	    self.DNN_locked = False
	    self._delta_buffer = []

    def _append_delta(self, curr_delta):	
        if curr_delta.is_mask_referenced:
	    self._last_delta_buffer = self._delta_buffer
	    self._delta_buffer = []
	    self._delta_buffer.append(curr_delta)
        else:
	    self._delta_buffer.append(curr_delta)
	    assert(len(self._delta_buffer) <= self._DELTA_BUFFER_MAX_SIZE)

    def _theta_correction(self, delta_theta):
	if not self._last_delta_buffer:
	    return delta_theta
	
	joined_delta_buffer = self._last_delta_buffer + self._delta_buffer
	window = joined_delta_buffer[-self._window_length-1:-1]
	window = [delta.dtheta_raw for delta in window]
	
	# Apply alpha trim filter to data
	print("Window before filter: ", window)
	window.sort()
	delta_theta = np.mean(np.array(window[self._alpha:-self._alpha]))
	print("Window after sort+cut: ", window[self._alpha:-self._alpha])
	
	return delta_theta

    @timer
    def _compute_centroid(self, mask, keypoints=np.array([])):
	if not keypoints.size == 0:
	    keypoints = keypoints.astype(np.int)
	    x = int(np.mean(keypoints[:,0]))
	    y = int(np.mean(keypoints[:,1]))
	else:
            M = cv2.moments(mask)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
        return (x,y)

    @timer
    def _manhattan_distance(self, center_point, mask_centroid):
        delta_x = mask_centroid[0] - center_point[0]
        delta_y = mask_centroid[1] - center_point[1]
        return (delta_x, delta_y)

    @timer
    def _compute_rotation(self, img, rotated_img, max_features=500, 
			  good_match_percent=0.10, keypoints=None, 
			  descriptors=np.array([])):

	# Compute ORB on input images and find the corresponding features
        orb = cv2.ORB_create(max_features)
	if not keypoints and descriptors.size==0:
            keypoints1, descriptors1 = orb.detectAndCompute(img, None)
	else:
	    keypoints1, descriptors1 = keypoints, descriptors
        keypoints2, descriptors2 = orb.detectAndCompute(rotated_img, None)
	
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        try:
	    matches = matcher.match(descriptors1, descriptors2, None)
	except Exception as e:
	    rospy.logwarn("CV2 ERROR! Skipping frame.")
	    delta_theta = 0
	    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	    imMatches = np.concatenate((img, rotated_img), axis=1)
	    H = np.zeros((3,3))	 
            return delta_theta, imMatches, keypoints2, descriptors2, H

	matches.sort(key=lambda x: x.distance, reverse=False)
        num_good_matches = int(len(matches) * good_match_percent) 
        matches = matches[:num_good_matches] # Only keep matches above good_match

	# Draw the matches on the img for visualization
        imMatches = cv2.drawMatches(img, keypoints1, rotated_img,
				    keypoints2, matches, None, matchColor=(0,255,0))

	# Create point correspondance lists
	points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

	# Find homography matrix based on good matches
	try:
            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
	except Exception as e:
	    H = np.zeros((3,3))	 

	# Replace translation parameters with centroid values for later extraction
	(x,y) = self._compute_centroid(img, points2)
	try: 
	    H[0,2] = x
	    H[1,2] = y
	except Exception as e:
	    H = np.zeros((3,3))	 
	    H[0,2] = x
	    H[1,2] = y
        
	# Since homography matrix H is defined as [R1 R2 t], the rotation about the
        # z axis can be computed with atan2(H12,H11)       
	try:
	    delta_theta = math.atan2(H[1,0],H[0,0])
	except Exception as e:
	    rospy.logwarn("Homography matrix emtpy.")
	    delta_theta = 0

	# Invert the rotation to account for the horizontal flip of the camera
        return -delta_theta, imMatches, keypoints2, descriptors2, H

    @timer
    def compute_control_mask(self, mask, mask_img):
	rospy.loginfo("Computing control based on mask data.")

	# Only consider images within the mask boundary 
	mask_img[np.where(mask == 0)] = 125

        # Get center pixel of frame
        center_pixel = (int(math.floor(mask.shape[1]/2.0)),
                        int(math.floor(mask.shape[0]/2.0)))

	# Compute centroid of detected mask
        centroid = self._compute_centroid(mask) 
        
	# Compute manhattan distance for delta_x and delta_y in PID control
        delta_x, delta_y = self._manhattan_distance(center_pixel, centroid)

        # Compute the \delta theta for control
        delta_theta_raw, img_matches, keypoints, descriptors, _ = self._compute_rotation(self._control_img, mask_img)
       
	delta_theta = self._theta_correction(delta_theta_raw)
	debug_img = img_matches
        
	# Create error vector \delta_x \delta_y \delta_theta and compute
        # output
	curr_delta = Delta(delta_x, delta_y,
			   delta_theta, delta_theta_raw,
			   is_mask_referenced=True)

	log = "Errors: delta_x: {}, delta_y: {}, delta_theta: {}"
	rospy.logdebug(log.format(curr_delta.dx, curr_delta.dy, curr_delta.dtheta*180/math.pi))
        
	# Return the errors plus the controller image visualization for debug	
	debug_img = compose_debug_img(debug_img, curr_delta)

	# Track errors sent as an internal state
	self._append_delta(curr_delta)
	
	rospy.loginfo("Control headings determined successfully!")
        return curr_delta, debug_img, keypoints, descriptors
    
    @timer
    def compute_control_image(self, img, mask, mask_img, keypoints, descriptors):
        rospy.loginfo("Computing control based on image to DNN mask correspondance.")
	if not keypoints:
	    pass
	debug_img = img

	# Only consider areas of the image in the mask
	mask_img[np.where(mask == 0)] = 0

        # Compute the \delta theta for control
        delta_theta, img_matches, keypoints, _, H = self._compute_rotation(mask_img, 
									img,
									keypoints=keypoints,
									descriptors=descriptors)
        
	# Reference the relative theta to mask determined theta and window
	delta_theta_raw = (delta_theta + (self._delta_buffer[0].errors[-1]))
	delta_theta = self._theta_correction(delta_theta_raw)

	debug_img = img_matches
       
	# Get center pixel of frame
        center_pixel = (int(math.floor(img.shape[1]/2.0)),
                        int(math.floor(img.shape[0]/2.0)))

	# Compute centroid of detected mask
        centroid = (int(H[0,2]), int(H[1,2]))

        # Compute manhattan distance for delta_x and delta_y in PID control
        delta_x, delta_y = self._manhattan_distance(center_pixel, centroid)

        # Create error vector \delta_x \delta_y \delta_theta and compute
        # output
	curr_delta = Delta(delta_x, delta_y,
			   delta_theta, delta_theta_raw,
			   is_mask_referenced=False)

	log = "Errors: delta_x: {}, delta_y: {}, delta_theta: {}"
	rospy.logdebug(log.format(curr_delta.dx, curr_delta.dy, curr_delta.dtheta*180/math.pi))

	if not self.DNN_locked:
	    curr_delta = Delta(0.0, 0.0, 0.0, is_mask_referenced=False)
	    rospy.logwarn("DNN lock lost...")
	    return curr_delta, debug_img

	debug_img = compose_debug_img(debug_img, curr_delta)

	# Track commands sent as an internal state
	self._append_delta(curr_delta)

	rospy.loginfo("Control headings determined successfully!")
        return curr_delta, debug_img


class Data_Handler(object):
    def __init__(self):
        # Data container
        class Data(object):
            def __init__(self):
                self.mask = None
                self.mask_image = None
                self.raw = None
        self.data = Data()

        # Mutii?... Mutexes?
        self.mask_available = False
        self.mask_image_available = False
        self.image_available = False

        # Private variables
        self._bridge = CvBridge()
        self._command_topic_pub = None
        self._debug_vis_pub = None

    def on_mask(self, img_msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(img_msg, "mono8")
        except CvBridgeError as e:
            print(e)

        if not self.mask_available:
            _, mask = cv2.threshold(cv_image,0.9,255,cv2.THRESH_BINARY)
	    # Check for mask
	    test = np.where(mask > 0)[0]
	    if(not test.size == 0):
            	rospy.loginfo("Received mask on radius/mrcnn/mask")
            	self.data.mask = mask
            	self.mask_available = True

    def on_img_mask(self, img_msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if not self.mask_image_available:
            rospy.logdebug("Received mask image on radius/mrcnn/img_mask")
	    mask_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.data.mask_image = mask_img
            self.mask_image_available = True

    def on_img(self, img_msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        if not self.image_available:
            rospy.logdebug("Received raw image on camera/raw")
            self.data.raw = cv_image
            self.image_available = True

    def publish_command(self, cmd, input_msg):
        """
        cmd: (6,) element array where the first three elements correspond to the
             formatted as linear_control_val and angular_control_val
	"""
        # Set up image message
        msg = Image()
        msg.header.stamp.secs = input_msg.header.stamp.secs
        msg.header.stamp.nsecs = input_msg.header.stamp.nsecs
        msg.header.frame_id = input_msg.header.frame_id
        msg.encoding = "32FC6";
        msg.width = 1
        msg.height = 1
        msg.step = msg.width * 6 * 4 # Magic numbers, don't change these

        # Easier to do the encoded message in c++ but it is still possible in
        # python using this round about method
        cmd = [axis_val for control_axis_type in cmd for axis_val in control_axis_type]
        cmd = np.array([[cmd]], dtype=np.float32)
        msg.data = struct.pack('ffffff', cmd[0,0,0], cmd[0,0,1], cmd[0,0,2],
                                         cmd[0,0,3], cmd[0,0,4], cmd[0,0,5])

    def publish_debug_visuals(self, debug_img):
        debug_img_msg = self._bridge.cv2_to_imgmsg(debug_img, "bgr8")
        self._debug_vis_topic_pub.publish(debug_img_msg)

    def set_command_pub(self, pub):
        self._command_topic_pub = pub

    def set_debug_visualizer_pub(self, pub):
        self._debug_vis_topic_pub = pub

def main():
    # Create data handler
    dh = Data_Handler()

    # Create landing controller
    # Setup control image
    img_path = os.path.join(os.getcwd(), cfg["CNTL_IMG_PATH"])
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
    (x,y) = cfg["CAMERA_DIMS"][0], cfg["CAMERA_DIMS"][1]
    img = cv2.resize(img, (x,y), interpolation= cv2.INTER_CUBIC)
    control_img = img
    lc = Lander(control_img)
	
    # Create directory to store the logs
    os.mkdir(LOGDIR_PATH)

    # Spin up landing controller
    if cfg["VERBOSITY"] == "DEBUG":
    	rospy.init_node('landing_controller', log_level=rospy.DEBUG)
    else:
	rospy.init_node('landing_conrtoller')
    rospy.loginfo("Starting Landing_Controller node...")

    # Subscribe to mask rcnn mask stream
    rospy.Subscriber("/radius/mask_rcnn/mask", Image, dh.on_mask)
    rospy.loginfo("Listening on radius/mask_rcnn/mask")

    # Subscribe to image mask channel to sychronize with the image the mask was drawn from
    rospy.Subscriber("radius/mask_rcnn/img_mask", Image, dh.on_img_mask)
    rospy.loginfo("Listening on radius/mask_rcnn/img_mask")

    # Subscribe to raw image in case no mask available
    rospy.Subscriber("camera/image_raw", Image, dh.on_img)
    rospy.loginfo("Listening on camera/image_raw")

    # Publish to network output heads
    #dh.set_command_pub(rospy.Publisher('/trails_dnn/network/output', Image,
    #                                   queue_size=20))

    # Publish to visualizer topic
    dh.set_debug_visualizer_pub(rospy.Publisher('/radius/landing_controller/visualization',
						Image, queue_size=20))
   
    # Waypoint commands must be faster than 2Hz
    rate = rospy.Rate(cfg["CONTROLLER"]["UPDATE_FREQ"])    
    keypoints = descriptors = None
    mask = None
    mask_img = None
    while not rospy.is_shutdown():
        if dh.mask_available and dh.mask_image_available:
	    lc.reset_lock_counter()
	    mask = dh.data.mask
	    mask_img = dh.data.mask_image
            curr_delta, debug_img, keypoints, descriptors = lc.compute_control_mask(mask, mask_img)

	    # Publish errors and save debug logs
	    cv2.imwrite(os.path.join(LOGDIR_PATH,"{}.png".format(time.time())),debug_img)
	    if cfg["VERBOSITY"] == "DEBUG":
	        cv2.imshow("Debug image", debug_img)
 	        cv2.waitKey(23)	

	    # TODO: change publishing to publish errors 
	    #dh.publish_command(ctrl_cmd, dh.data.raw)
            dh.publish_debug_visuals(debug_img)
            dh.mask_available = False
	    dh.mask_image_available = False

	elif not lc.DNN_locked:
	    keypoints = None 
	    dh.mask_image_available = False

	elif dh.image_available and lc.DNN_locked: 
	    if dh.mask_image_available:
		lc.increment_no_lock_counter()
	    	dh.mask_image_available = False 

	    if not keypoints == None and lc.DNN_locked:
		raw = dh.data.raw
	    	curr_delta, debug_img = lc.compute_control_image(raw, mask, mask_img,
							        keypoints, descriptors)	
		# Publish commands and save debug logs
	    	cv2.imwrite(os.path.join(LOGDIR_PATH,"{}.png".format(time.time())),debug_img)
	    	if cfg["VERBOSITY"] == "DEBUG":
	            cv2.imshow("Debug image", debug_img)
 	            cv2.waitKey(23)	

		dh.publish_debug_visuals(debug_img)
		# TODO: change to publish errors
		#dh.publish_command(ctrl_cmd, dh.data.raw) 

	    dh.image_available = False

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
