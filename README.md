# Autonomous-Fighting-Robot-Project

This repository outlines the code that was used for ES327 - Individual Project. 

Order of Implementation: 
1. calibrate_camera.py uses "Camera Calibration Images" to retrieve intrinstic camera parameters.
2. ArUco_Marker_Generation.py produces ArUco markers to be printed.
3. robot_server.py is uploaded on the ESP32. 
4. Then: run PHASE_1.py - finite state machine.
5.    or run PHASE_2.py - direct proportional drive.
6.    or run PHASE_3.py - PID drive. 
