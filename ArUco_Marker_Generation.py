import cv2

# Generate and save marker ID 0 from DICT_4X4_1000
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
marker_img  = cv2.aruco.generateImageMarker(aruco_dict, 0, 500)
cv2.imwrite("test_marker_4x4_1000_id0.png", marker_img)
print("Saved test_marker_4x4_1000_id0.png — print this and test")