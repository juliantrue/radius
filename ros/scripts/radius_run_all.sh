xterm -e watch -n 0.5 sensors &
./run_MAVROS.sh > /dev/null &
./run_gscam.sh > /dev/null &
./run_mrcnn.sh &
./run_landing_controller.sh 

wait
echo Processes terminated.
