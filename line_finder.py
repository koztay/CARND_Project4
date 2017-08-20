import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
polyfit fonksiyonu katsayıları veriyor, o eğriyi çizebilmek için gerekli katsayılar...

"""

WARPED_IMAGES_FOLDER = "output_images/perspective_transform/"
# straight_lines1_undistorted_binary_perspective.jpg
# test2_undistorted_binary_perspective.jpg

# img = cv2.imread(WARPED_IMAGES_FOLDER + "straight_lines1_undistorted_binary_perspective.jpg")
#  5880.21533824 m 2786.43255789 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "straight_lines2_undistorted_binary_perspective.jpg")
# 28954.3091061 m 2399.61986047 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "test1_undistorted_binary_perspective.jpg")
# 600.166747012 m 761.316145614 m

img = cv2.imread(WARPED_IMAGES_FOLDER + "test2_undistorted_binary_perspective.jpg")
# 619.063231512 m 431.589715181 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "test3_undistorted_binary_perspective.jpg")
# 809.544675531 m 749.205320075 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "test4_undistorted_binary_perspective.jpg")
# 891.558331479 m 303.623039159 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "test5_undistorted_binary_perspective.jpg")
# 1419.54206488 m 254.583950579 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "test6_undistorted_binary_perspective.jpg")
# 2951.6956695 m 419.285377061 m

# img = cv2.imread(WARPED_IMAGES_FOLDER + "warped-example.jpg")
# 1046.03640972 m 377.992604342 m



mark_lines(img)
