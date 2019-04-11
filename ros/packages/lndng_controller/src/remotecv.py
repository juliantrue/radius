import cv2
import numpy as np
import sys
import subprocess
import shlex
import os

gstCommand = None

def initialize(host, port, bitrate=2048, w=640, h=480):
    global gstCommand
    if os.name == "posix":
        #args = shlex.split(('gst-launch-1.0 fdsrc ! videoparse format="i420" width={} height={}' +
        #' ! x264enc speed-preset=1 tune=zerolatency bitrate={}' +
        #' ! rtph264pay config-interval=1 pt=96 ! udpsink host={} port={}').format(
        #w, h, bitrate, host, port))
        args = shlex.split(('gst-launch-1.0 fdsrc ! videoparse format="i420" width={} height={} ! jpegenc ! rtpjpegpay ! udpsink host={} port={}').format(w, h, host, port))
	gstCommand = subprocess.Popen(args, stdin=subprocess.PIPE)

def imshow(name, img):
    if gstCommand:
        gstCommand.stdin.write(cv2.cvtColor(img, cv2.COLOR_RGB2YUV_I420))
    else:
        cv2.imshow(name,img)

def waitKey(delay):
    if os.name == "posix":
        return -1
    return cv2.waitKey(delay)
