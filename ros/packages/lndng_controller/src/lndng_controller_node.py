#!/usr/bin/python
from __future__ import division
from __future__ import print_function
# Ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Python libs
import sys, os, time, struct, math
from multiprocessing import Lock
import numpy as np
import cv2


class Lander(object):
    def __init__(self, control_img):
        self._control_mask = control_img
        self._cmd_tracker = []
        self._CMD_TRACKER_MAX_SIZE = 16
        self._alpha_smoothing = 1.0
        self._controller = self.PI(4.0,0.1,0.01)

    class PI(object):
        def __init__(self, sampling_frequency, Kp, Ki):
            self._sampling_frequency = sampling_frequency
            self._Kp = Kp
            self._Ki = Ki
            self._error_tracker = []
            self._ERROR_TRACKER_MAX_SIZE = 64

        def compute_output(self, error):
            if len(self._error_tracker) < self._ERROR_TRACKER_MAX_SIZE:
                self._error_tracker.append(error)
            else:
                self._error_tracker = self._error_tracker[1:]
                self._error_tracker.append(error)
            output = self._proportional() + self._integral()
            return output

        def _proportional(self):
            output = self._Kp * self._error_tracker[-1]
            return output

        def _integral(self):
            delta_t = (1/self._sampling_frequency)*np.ones((len(self._error_tracker),))
            # Discrete time integral = \Sigma^t_0 error*delta_t
            integral = np.array(self._error_tracker).dot(delta_t)
            output = self._Ki * integral
            return output

    def _compute_centroid(self, mask):
        M = cv2.moments(mask)
        x = int(M["m10"] / M["m00"])
        y = int(M["m01"] / M["m00"])
        return (x,y)

    def _manhattan_distance(self, center_point, mask_centroid):
        delta_x = mask_centroid[0] - center_point[0]
        delta_y = mask_centroid[1] - center_point[1]
        return (delta_x, delta_y)

    def _compute_rotation(self, known_mask, mask):
        # Find contours of detected mask and fit box and line to it
        des = cv2.bitwise_not(mask)
        _, contours, _ = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[-1]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

        # Find contours of known mask and fit box and line to it
        des = cv2.bitwise_not(known_mask)
        _, contours, _ = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        known_cnt = contours[-1]
        rect = cv2.minAreaRect(known_cnt)
        kbox = cv2.boxPoints(rect)
        kbox = np.int0(kbox)
        [kvx,kvy,kx,ky] = cv2.fitLine(known_cnt, cv2.DIST_L2,0,0.01,0.01)

        # Compute \delta angle
        vec = np.squeeze(np.array([vx,vy,x,y]))
        kvec = np.squeeze(np.array([kvx,kvy,kx,ky]))

        k_ang = math.atan2(kvec[1],kvec[0])
        ang = math.atan2(vec[1],vec[0])
        delta_theta = ang - k_ang

        return delta_theta, vec, kvec, box, kbox

    def compute_control(self, mask):
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        debug_img = np.zeros(mask.shape,dtype=np.uint8)

        # Check if mask is present in image. If not, pass
        temp = np.where(mask > 0)
        if temp[0].size == 0:
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), debug_img

        # Get center pixel of frame
        center_pixel = (int(math.floor(mask.shape[1]/2.0)),
                        int(math.floor(mask.shape[0]/2.0)))

        centroid = self._compute_centroid(mask) # Compute centroid of detected mask

        # Compute manhattan distance for delta_x and delta_y in PID control
        delta_x, delta_y = self._manhattan_distance(center_pixel, centroid)

        # Compute the \delta theta for PID control
        delta_theta, vec, kvec, box, kbox = self._compute_rotation(self._control_mask, mask)

        # Create error vector \delta_x \delta_y \delta_theta and compute PID
        # output
        error = (delta_x, delta_y, delta_theta*180 / math.pi)
        error_mag = math.sqrt(delta_x**2 + delta_y**2)
        error_ang = math.atan2(delta_x, delta_y)

        # Log to stdout
        #rospy.loginfo("Delta_x={.4f}, Delta_y={.4f}, Delta_theta={.4f}".format(error[0], error[1], error[2]))

        # TODO: clean this block of code up.... its really gross
        debug_img = mask
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        shape = debug_img.shape
        rows,cols = debug_img.shape[:2]
        lefty = int((-kvec[2]*kvec[1]/kvec[0]) + kvec[3])
        righty = int(((cols-kvec[2])*kvec[1]/kvec[0])+kvec[3])
        cv2.line(debug_img,(cols-1,righty),(0,lefty),(0,255,0),2)
        lefty = int((-vec[2]*vec[1]/vec[0]) + vec[3])
        righty = int(((cols-vec[2])*vec[1]/vec[0])+vec[3])
        cv2.line(debug_img,(cols-1,righty),(0,lefty),(0,255,0),2)
        cv2.drawContours(debug_img,[box],0,(0,0,255),2)
        cv2.circle(debug_img, centroid, 5, (0,0,255), -1)
        cv2.circle(debug_img, center_pixel, 5, (0,0,255), -1)
        cv2.line(debug_img, center_pixel, (center_pixel[0]+delta_x,
                    center_pixel[1]),(255,0,0),1)
        cv2.line(debug_img, (center_pixel[0]+delta_x,center_pixel[1]),
                    (center_pixel[0]+delta_x,center_pixel[1]+delta_y),(255,0,0),1)

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
	cmd = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cmd, debug_img


class Data_Handler(object):
    def __init__(self):
        class Data(object):
            def __init__(self):
                self.img = None
                self.raw = None
        self.data = Data()
        self.data_available = False
        self._bridge = CvBridge()
        self._command_topic_pub = None
        self._debug_vis_pub = None

    def on_img(self, img_msg):
        try:
            cv_image = self._bridge.imgmsg_to_cv2(img_msg, "mono8")
        except CvBridgeError as e:
            print(e)

        if not self.data_available:
            rospy.loginfo("Mask available!")
            _, cv_image = cv2.threshold(cv_image,0.9,255,cv2.THRESH_BINARY)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            self.data.raw = img_msg
	    self.data.img = cv_image
	    self.data_available = True

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
    img_path = os.path.join(os.getcwd(), 'src/lndng_controller/src/control_img/elephant.bmp')
    control_img = cv2.imread(img_path, 0)
    lc = Lander(control_img)

    rospy.init_node('landing_controller')
    rospy.loginfo("Starting Landing_Controller node...")

    # Subscribe to mask rcnn mask stream
    rospy.Subscriber("/radius/mask_rcnn/mask", Image, dh.on_img)

    # Publish to network output heads
    #dh.set_command_pub(rospy.Publisher('/trails_dnn/network/output', Image,
    #                                   queue_size=20))

    dh.set_debug_visualizer_pub(rospy.Publisher('/radius/landing_controller/visualization', Image,
                                       queue_size=20))

    discount_factor = 2
    rate = rospy.Rate(4) # Waypoint commands must be faster than 2Hz
    while not rospy.is_shutdown():
        if(dh.data_available): # Mask available to determine control vectors
	    ctrl_cmd, debug_img = lc.compute_control(dh.data.img)
            #dh.publish_command(ctrl_cmd, dh.data.raw)
            dh.publish_debug_visuals(debug_img)
            dh.data_available = False
	"""
        else: # Use previous image to send commands with discount factor
	    ctrl_cmd, debug_img = lc.compute_control(dh.data.img)
            for i,cmd in enumerate(ctrl_cmd):
                ctrl_cmd[i] /= discount_factor
            dh.publish_command(ctrl_cmd, dh.data.raw)
            dh.publish_debug_visuals(debug_img)
	"""
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
