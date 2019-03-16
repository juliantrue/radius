#!/usr/bin/python
from __future__ import division
from __future__ import print_function
# Ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Python libs
import sys, os, time, struct, math
import functools
from multiprocessing import Lock
import numpy as np
import cv2


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


class Lander(object):
    def __init__(self, control_img):
        self._control_img = control_img
        self._cmd_tracker = []
        self._CMD_TRACKER_MAX_SIZE = 16
        self._alpha_smoothing = 1.0

    @timer
    def _compute_centroid(self, mask, keypoints=None):
	if keypoints:
	    kpt_img = np.zeros(mask.shape[-2:-1])
	    points = np.array([(keypoint.pt[1],keypoint.pt[0]) for keypoint in keypoints], dtype=np.int)
	    kpt_img[points] = 255 
            M = cv2.moments(kpt_img)
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

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
    def _compute_rotation(self, img, rotated_img, max_features=500, good_match_percent=0.15, keypoints=None, descriptors=None):

	# Compute ORB on input images and find the corresponding features
        orb = cv2.ORB_create(max_features)
        keypoints1, descriptors1 = orb.detectAndCompute(img, None)
	if keypoints and not descriptors.size == 0:
	    keypoints2, descriptors2 = keypoints, descriptors 	
	else:
            keypoints2, descriptors2 = orb.detectAndCompute(rotated_img, None)

        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        numGoodMatches = int(len(matches) * good_match_percent) # Only keep matches above good_match
        matches = matches[:numGoodMatches]

        # Draw the matches on the img for visualization
        imMatches = cv2.drawMatches(img, keypoints1, rotated_img, keypoints2, matches, None)

	cv2.namedWindow("test")
	cv2.imshow("test",imMatches)
	cv2.waitKey(23)

        # Create the points list of only good points for homography matrix solving
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Solve for the homography matrix and compute the theta_z euler angle from it
        H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        # Since homography matrix H is defined as [R1 R2 t], the rotation about the
        # z axis can be computed with atan2(H12,H11)
        delta_theta = math.atan2(H[1,0],H[0,0])

        return -delta_theta, imMatches, keypoints2, descriptors2

    @timer
    def compute_control_mask(self, mask, mask_img):
	rospy.loginfo("Computing control based on mask data.")

	# Only consider images within the mask boundary 
	mask_img[np.where(mask == 0)] = 0

        # Get center pixel of frame
        center_pixel = (int(math.floor(mask.shape[1]/2.0)),
                        int(math.floor(mask.shape[0]/2.0)))

        centroid = self._compute_centroid(mask) # Compute centroid of detected mask

        # Compute manhattan distance for delta_x and delta_y in PID control
        delta_x, delta_y = self._manhattan_distance(center_pixel, centroid)

        # Compute the \delta theta for control
        delta_theta, img_matches, keypoints, descriptors = self._compute_rotation(self._control_img, mask_img)
        debug_img = img_matches

        # Create error vector \delta_x \delta_y \delta_theta and compute
        # output
        error = (delta_x, delta_y, delta_theta*180 / math.pi)
        error_mag = math.sqrt(delta_x**2 + delta_y**2)
        error_ang = math.atan2(delta_x, delta_y)

	rospy.logdebug("Errors: delta_x: {}, delta_y: {}, delta_theta: {}".format(error[0], error[1], error[2]))


        ############ Compute control from errors ##############################

        #error_mag = self._controller.compute_output(error_mag)
        linear_control_val = error_mag*math.cos(error_ang)
        angular_control_val = error_mag*math.sin(error_ang)

        # Do Exponential smoothing on linear and angular commands
        if len(self._cmd_tracker) > 1:
            linear_control_val = linear_control_val*(1-self._alpha_smoothing) + self._cmd_tracker[-1]*self._alpha_smoothing

        # Append commands to buffer to use in future steps
        """
        if len(self._cmd_tracker) <= self._CMD_TRACKER_MAX_SIZE:
            self._cmd_tracker.append((linear_control_val, angular_control_val, yaw_control_val,
                    altitude_control_val, 0.0, 0.0))
        else:
            self._cmd_tracker = self._cmd_tracker[1:]
            self._cmd_tracker.append((linear_control_val, angular_control_val, yaw_control_val,
                    altitude_control_val, 0.0, 0.0))

        """
        # Return the command plus the controller image visualization for debug
        #return (linear_control_val, angular_control_val, yaw_control_val,
        #        altitude_control_val, 0.0, 0.0), debug_img

	rospy.loginfo("Control headings determined successfully!")
        cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cmd, debug_img, keypoints, descriptors
    
    @timer
    def compute_control_image(self, img, keypoints, descriptors):
        rospy.loginfo("Computing control based on image to DNN mask correspondance.")
	debug_img = img

        # Get center pixel of frame
        center_pixel = (int(math.floor(img.shape[1]/2.0)),
                        int(math.floor(img.shape[0]/2.0)))

	# Compute centroid of detected mask
        centroid = self._compute_centroid(img, keypoints=keypoints)

        # Compute manhattan distance for delta_x and delta_y in PID control
        delta_x, delta_y = self._manhattan_distance(center_pixel, centroid)

        # Compute the \delta theta for control
        delta_theta, img_matches, keypoints, descriptors = self._compute_rotation(self._control_img, img)
        debug_img = img_matches

        # Create error vector \delta_x \delta_y \delta_theta and compute
        # output
        error = (delta_x, delta_y, delta_theta*180 / math.pi)
        error_mag = math.sqrt(delta_x**2 + delta_y**2)
        error_ang = math.atan2(delta_x, delta_y)

	rospy.logdebug("Errors: delta_x: {}, delta_y: {}, delta_theta: {}".format(error[0], error[1], error[2]))


        ############ Compute control from errors ##############################

        linear_control_val = error_mag*math.cos(error_ang)
        angular_control_val = error_mag*math.sin(error_ang)

	cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	return cmd, debug_img	


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

        #print(msg)
        # Test that the input looks like the output
        #data = bytearray(list(msg.data))
        #print(struct.unpack('<%df' % (len(data) / 4), data))

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
    # Initialize with a control binary mask from the COCO dataset. Use for
    # reference in control. One bmp image should exist in control_img dir
    img_path = os.path.join(os.getcwd(), 'src/lndng_controller/src/control_img/elephant.png')
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    ret, img_mask = cv2.threshold(img[:,:,3],0,255,cv2.THRESH_BINARY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not img_mask[i,j] == 255:
                img[i,j] = 0

    img = cv2.resize(img, (640,480), interpolation= cv2.INTER_CUBIC)
    control_img = img
    lc = Lander(control_img)

    # Spin up landing controller
    rospy.init_node('landing_controller', log_level=rospy.DEBUG)
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
    dh.set_debug_visualizer_pub(rospy.Publisher('/radius/landing_controller/visualization', Image, queue_size=20))

    rate = rospy.Rate(4) # 250ms Waypoint commands must be faster than 2Hz
    keypoints = descriptors = None
    while not rospy.is_shutdown():
        if dh.mask_available and dh.mask_image_available:
            ctrl_cmd, debug_img, keypoints, descriptors = lc.compute_control_mask(dh.data.mask, dh.data.mask_image)
	    #dh.publish_command(ctrl_cmd, dh.data.raw)
            dh.publish_debug_visuals(debug_img)
            dh.mask_available = False
	    dh.mask_image_available = False

	elif dh.image_available:
	    if dh.mask_image_available:
	    	dh.mask_image_available = False
	    
	    if not keypoints == None:
	    	ctrl_cmd, debug_img = lc.compute_control_image(dh.data.raw, keypoints, descriptors)
		dh.publish_debug_visuals(debug_img)
		#dh.publish_command(ctrl_cmd, dh.data.raw)
	   
	    dh.image_available = False

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
