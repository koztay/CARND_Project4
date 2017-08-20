import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import pickle
import matplotlib.pyplot as plt



# imgpoints, objpoints = calculate_calibration_points(6, 9, "camera_cal/calibration*.jpg")

# # images = glob.glob("camera_cal/calibration*.jpg")


# undistort all test images
# undistort_test_images(mtx, dist)

# binary image
# image_path = "test_images/calibration.jpg"  # grabbed the first image
# img = mpimg.imread(image_path)
# plt.imshow(img)
# plt.show()
# binary_img = binary_image(img)
# Before converting pic to uint8, you need to multiply it by 255 to get the correct range
# cv2.imwrite("output_images/binary_images/binary.jpg", binary_img*255)


# perspective transform each image
def birds_eye_view():
    """
    # Four sources coordinates
    src = np.float32([[850, 320],
                      [865, 450],
                      [533, 350],
                      [535, 210]])

    # Four desired coordinates
    dst = np.float32([[870, 240],
                      [870, 370],
                      [520, 370],
                      [520, 240]])
    """

    s_point1 = [302, 650]
    s_point2 = [596, 456]
    s_point3 = [708, 464]
    s_point4 = [1002, 650]

    d_point1 = [400, 650]
    d_point2 = [435, 250]
    d_point3 = [870, 250]
    d_point4 = [880, 650]
    #
    # s_point1 = [364, 613]
    # s_point2 = [551, 481]
    # s_point3 = [706, 462]
    # s_point4 = [940, 613]
    #
    # d_point1 = [400, 613]
    # d_point2 = [400, 481]
    # d_point3 = [900, 462]
    # d_point4 = [900, 613]

    # s_point1 = [277, 680]
    # s_point2 = [618, 433]
    # s_point3 = [658, 433]
    # s_point4 = [1041, 676]
    #
    # d_point1 = [346, 718]
    # d_point2 = [346, 5]
    # d_point3 = [934, 5]
    # d_point4 = [934, 716]

    dist_pickle = pickle.load(open("calibration_mtx_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    src = np.float32([s_point1, s_point2, s_point3, s_point4])
    dst = np.float32([d_point1, d_point2, d_point3, d_point4])

    images = glob.glob("test_images/*.jpg")
    for img_path in images:
        img = cv2.imread(img_path)

        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        binary_img = binary_image(undistorted)
        warped_img = warper(binary_img, src, dst)
        filename = img_path.split("/")[1].split(".")[0]  # removed path and after extension
        cv2.imwrite("output_images/warped_images/{}_warped.jpg".format(filename), warped_img*255)

# birds_eye_view()
# img = mpimg.imread("test_images/test1.jpg")
# binary_img = binary_image(img)
# cv2.imwrite("output_images/binary_images/binary.jpg", binary_img*255)
#
# img = mpimg.imread("output_images/test_images/straight_lines1_undistorted.jpg")
# # plt.imshow(img)
# # plt.show()
# # print(img_size)
# img_size = (1280, 720)
#
# xff_top_left = 54
# xff_top_right = 55
# xff_bottom_left = 25
# xff_bottom_right = 45
# yff = 95
#
# # best found
# # xff = 68
# # yff = 98
#
# # source image points
#
#
# src = np.float32(
#     [[(img_size[0] / 2) - xff_top_left, img_size[1] / 2 + yff],   # top_left
#      [((img_size[0] / 6) - xff_bottom_left), img_size[1]],        # bottom_left
#      [(img_size[0] * 5/6) + xff_bottom_right, img_size[1]],        # bottom_right
#      [(img_size[0] / 2 + xff_top_right), img_size[1] / 2 + yff]])  # top_right
# dst = np.float32(
#     [[(img_size[0] / 4), 1],
#      [(img_size[0] / 4), img_size[1]],
#      [(img_size[0] * 3 / 4), img_size[1]],
#      [(img_size[0] * 3 / 4), 1]])
#
# warper(img, src, dst)