gst-launch-1.0 udpsrc port=5800 caps=application/x-rtp,payload=96 ! rtph264depay ! avdec_h264 ! xvimagesink
