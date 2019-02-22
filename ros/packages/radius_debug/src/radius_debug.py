#!/usr/bin/python
# Ros stuff
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# Python libs
import sys, os
import numpy as np
import cv2

class Data_Handler(object):
    def __init__(self):
        self.mask_available = False
        self.masked_img_available = False
        self.ctrl_img_available = False
        self.mask = None
        self.masked_img = None
        self.ctrl_img = None
        self._bridge = CvBridge()

    def on_img(self, img_msg):
        try:
            img = self._bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Save data as interal state
        self.masked_img = img
        self.masked_img_available = True

    def on_mask(self, mask):
        try:
            mask = self._bridge.imgmsg_to_cv2(mask, "mono8")
        except CvBridgeError as e:
            print(e)

        # Threshold and convert to three channel img for mask visibility
        _, mask = cv2.threshold(mask,0.9,255,cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Save data as interal state
        self.mask = mask
        self.mask_available = True

    def on_ctrl(self, ctrl_img):
        try:
            ctrl_img = self._bridge.imgmsg_to_cv2(ctrl_img, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Save data as interal state
        self.ctrl_img = ctrl_img
        self.ctrl_img_available = True


def main():
    # Start node and data handler
    dh = Data_Handler()
    rospy.init_node('Radius_Debug')
    rospy.loginfo("Starting debugging node...")

    # Subscribe to debug topics
    rospy.Subscriber('radius/mask_rcnn/img_mask', Image, dh.on_img)
    rospy.Subscriber('radius/mask_rcnn/mask', Image, dh.on_mask)
    rospy.Subscriber('radius/landing_controller/visualization', Image, dh.on_ctrl)
    rospy.loginfo("Listening on topic radius/mask_rcnn/img_mask")
    rospy.loginfo("Listening on topic radius/mask_rcnn/mask")
    rospy.loginfo("Listening on topic radius/landing_controller/visualization")


    rate = rospy.Rate(4)
    while not rospy.is_shutdown():
        if(dh.mask_available and dh.masked_img_available and dh.ctrl_img_available):
            mask = dh.mask
            masked_img = dh.masked_img
            ctrl_img = dh.ctrl_img
            debugging_img = np.concatenate((masked_img, mask, ctrl_img), axis=1)
            cv2.imshow("Radius Output",debugging_img)
            cv2.waitKey(3)
        rate.sleep()
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
