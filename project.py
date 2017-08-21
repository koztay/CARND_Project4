import cv2
import glob
import numpy as np
import pickle

from moviepy.editor import VideoFileClip

IMG_SIZE = (1280, 720)

XF_TOP_LEFT = 54
XF_TOP_RIGHT = 55
XF_BOTTOM_LEFT = 25
XF_BOTTOM_RIGHT = 45
YF = 95

SRC = np.float32(
    [[(IMG_SIZE[0] / 2) - XF_TOP_LEFT, IMG_SIZE[1] / 2 + YF],  # top_left
     [((IMG_SIZE[0] / 6) - XF_BOTTOM_LEFT), IMG_SIZE[1]],  # bottom_left
     [(IMG_SIZE[0] * 5 / 6) + XF_BOTTOM_RIGHT, IMG_SIZE[1]],  # bottom_right
     [(IMG_SIZE[0] / 2 + XF_TOP_RIGHT), IMG_SIZE[1] / 2 + YF]])  # top_right
DST = np.float32(
    [[(IMG_SIZE[0] / 4), 0],
     [(IMG_SIZE[0] / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), 0]])

# Define conversions in x and y from pixels space to meters
YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

NUM_ITERATIONS_TO_KEEP = 5
NUM_IMAGES_TO_KEEP = 2

def calculate_calibration_points(num_rows, num_cols, glob_images_path):
    """
    Calculates calibration points required for valibrating the camera and returns them
    :param num_rows: number of rows for calibration image_set
    :param num_cols: number of columns for calibration images_set
    :param glob_images_path: path of images like "camera_cal/calibration*.jpg"
    :return: imgpoints (coordinates of point as pixel values) , objpoints (coordinates as square index numbers)
    """
    # Read all images by glob api
    images = glob.glob(glob_images_path)  # star koyduğumuz yere 1, 2, 3 diyerek okuyor.

    # Arrays to store object points and image points from the all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points, like (0, 0 ,0), (1, 0, 0), (2, 0, 0) ....., (7, 5, 0)
    objp = np.zeros((num_cols * num_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2)  # x, y coordinates

    for fname in images:
        # read each image
        img = cv2.imread(fname)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        """
        # Cannot calculate points for the following images:
        Could not find chessboardcorners for : camera_cal/calibration1.jpg
        Could not find chessboardcorners for : camera_cal/calibration4.jpg
        Could not find chessboardcorners for : camera_cal/calibration5.jpg
        """
        ret, corners = cv2.findChessboardCorners(gray, (num_cols, num_rows), None)

        # If corners are found, add object points, image points
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            print("Could not find chessboardcorners for : {}".format(fname))

    # Save objpoints and imgpoints always overwrite the calibration data
    dist_pickle = dict()
    dist_pickle["objpoints"] = objpoints
    dist_pickle["imgpoints"] = imgpoints
    pickle.dump(dist_pickle, open("calibration_points_pickle.p", "wb"))

    return imgpoints, objpoints


def calculate_mtx_and_dist_points(imgpoints, objpoints, img_size):
    """
    Calibrates camera and writes undistorted image (not warped!!!)  to disk
    :param imgpoints: imgpoints returned from calculate_calibration_points() function
    :param objpoints: objpoints returned from calculate_calibration_points() function
    :param img_size: image size for calibration parameters
    :return: True
    """

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = dict()
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calibration_mtx_dist_pickle.p", "wb"))

    return mtx, dist


def undistort_images(mtx, dist, images):
    """
    undistorts images in a folder read by glob api
    :param mtx: calibration matrix parameter
    :param dist:  calibration dist parameter
    :param images: images = glob.glob("camera_cal/*.jpg")
    :return: undistorted image paths as an array like glob api returns.
    """
    undistorted_image_paths = []
    # Use the OpenCV undistort() function to remove distortion
    for image in images:
        img = cv2.imread(image)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        filename = image.split("/")[1].split(".")[0]  # removed path and after extension
        undist_img_path = "output_images/00_undistorted_images/{}.jpg".format(filename)
        undistorted_image_paths.append(undist_img_path)
        cv2.imwrite(undist_img_path, undist)

    return undistorted_image_paths


def write_to_disk(img, image_path, write_directory, is_binary=False):
    image_name = image_path.split("/")[-1].split(".")[0]
    new_image_path = write_directory + image_name + ".jpg"
    if write_to_disk:
        if is_binary:
            cv2.imwrite(new_image_path, img*255)
        else:
            cv2.imwrite(new_image_path, img)


def binary_image(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Before converting pic to uint8, you need to multiply it by 255 to get the correct range
    # cv2.imwrite("output_images/binary_images/sx.jpg", sxbinary*255)

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Before converting pic to uint8, you need to multiply it by 255 to get the correct range
    # cv2.imwrite("output_images/binary_images/s.jpg", s_binary*255)
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(sxbinary >= 1) | (s_binary >= 1)] = 1

    return combined_binary
    # return color_binary


def warper(img, **kwargs):
    src = kwargs.get("src")
    dst = kwargs.get("dst")
    inv = kwargs.get("inv")

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    # print(img_size)
    # plt.imshow(img)
    # # source image points
    # plt.plot(src[0][0], src[0][1], '.')  # top left
    # plt.plot(src[1][0], src[1][1], '.')  # bottom left
    # plt.plot(src[2][0], src[2][1], '.')  # bottom right
    # plt.plot(src[3][0], src[3][1], '.')  # top right
    #
    # # source image points
    # plt.plot(dst[0][0], dst[0][1], '.')  # top left
    # plt.plot(dst[1][0], dst[1][1], '.')  # bottom left
    # plt.plot(dst[2][0], dst[2][1], '.')  # bottom right
    # plt.plot(dst[3][0], dst[3][1], '.')  # top rigt
    # plt.show()

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    if inv:
        matrix = Minv
    else:
        matrix = M

    warped = cv2.warpPerspective(img, matrix, img_size, flags=cv2.INTER_LINEAR)  # keep same size as input image

    return warped


def calibration_pipeline():
        # step 1: calculate and save calibration points
        img_p, obj_p = calculate_calibration_points(9, 6, "camera_cal/calibration*.jpg")

        # step 2: calulate image size for mtx and dist calculation
        img = cv2.imread("camera_cal/calibration1.jpg")
        img_size = (img.shape[1], img.shape[0])

        # step 3: calculate mtx and dist parameters:
        mtx, dist = calculate_mtx_and_dist_points(img_p, obj_p, img_size)

        # step 4: undistort and save calibration images used for calibration
        images = glob.glob("camera_cal/*.jpg")
        undistort_images(mtx, dist, images)


def measure_curvature(ploty=None,
                      left_fit=None,
                      right_fit=None,
                      leftx=None,
                      rightx=None):

    # print(ploty.shape)
    # print(leftx.shape)
    # print(rightx.shape)

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    # print(left_curverad, right_curverad)
    # Example values: 1926.74 1908.48

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = YM_PER_PIX  # meters per pixel in y dimension
    xm_per_pix = XM_PER_PIX  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    # print("left_fit_cr", left_fit_cr)
    # print("len(left_fit_cr)", len(left_fit_cr))

    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m

    curverad = (left_curverad + right_curverad) / 2

    return curverad


def calculate_vehicle_pos(lane_center):
    difference_from_center = 640 - lane_center
    # print("difference_from_center", difference_from_center)
    distance_in_meters = difference_from_center * XM_PER_PIX
    if distance_in_meters == 0:  # Vehicle is in center
        position = "Vehicle is in center."
    elif distance_in_meters > 0:
        position = "Vehicle is {0:.2f} m right of center.".format(distance_in_meters)
    else:
        position = "Vehicle is {0:.2f} m left of center.".format(-distance_in_meters)
    return position


class ImgPipeLine:
    def __init__(self):

        self.curvature = []  # This is for image pipeline
        self.filled_lane = []
        self.lane_centers = []

        # x values of the last n fits of the line
        self.left_fit = []
        self.right_fit = []

        self.ploty = []
        self.left_fitx = []
        self.right_fitx = []
        # self.fill_lane_area(result, ploty, left_fitx, right_fitx)

        # self.leftx = None
        # self.rightx = None

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        # self.recent_xfitted = []

        # average x values of the fitted line over the last n iterations
        # self.bestx = None
        #
        # # polynomial coefficients averaged over the last n iterations
        # self.best_fit = None
        #
        # # polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]

        # # radius of curvature of the line in some units
        # self.radius_of_curvature = None
        #
        # # distance in meters of vehicle center from the line
        # self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

    def calculate_mean_fit(self):

        mean_left = np.mean(self.left_fit, axis=0)
        mean_right = np.mean(self.right_fit, axis=0)
        # print("mean_left, mean_right", mean_left, mean_right)
        ploty = self.ploty
        # print("self.ploty", self.ploty)
        left_fitx = mean_left[0] * ploty ** 2 + mean_left[1] * ploty + mean_left[2]
        right_fitx = mean_right[0] * ploty ** 2 + mean_right[1] * ploty + mean_right[2]
        return left_fitx, right_fitx

    def fill_lane_area(self, img, ploty, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warped_zero = np.zeros_like(img)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])

        # print(pts_left[0][719][0])
        # print(pts_right[0][0][0])

        lane_center = (left_fitx[719] + right_fitx[719]) / 2
        self.lane_centers.append(lane_center)

        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        filled = cv2.fillPoly(warped_zero, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        # newwarp = cv2.warpPerspective(warp_zero, Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        # Combine the result with the original image
        # result = cv2.addWeighted(img, 1, filled, 0.3, 0)
        return filled

    def mark_lines(self, img):
        binary_warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # binary_warped = np.copy(img)

        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)
        # plt.show()

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        # print("midpoint", midpoint)  # 1280 / 2 = 680

        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # print("leftx_base", leftx_base)  # 334
        # print("rightx_base", rightx_base)  # 948

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)  # height / 9

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        # print("nonzero", nonzero)
        # print("nonzero_shape", nonzero.shape)  # 2 adet array var içinde x arrayi ve y arrayi

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # print("nonzeroy", nonzeroy)
        # print("nonzeroy_shape", nonzeroy.shape)
        # print("nonzerox", nonzerox)
        # print("nonzerox_shape", nonzerox.shape)
        """
        nonzero (array([  0,   0,   0, ..., 719, 719, 719]), array([   0,   13,   14, ..., 1002, 1005, 1007]))
        nonzeroy [  0   0   0 ..., 719 719 719]
        nonzeroy_shape (108419,)
        nonzerox [   0   13   14 ..., 1002 1005 1007]
        nonzerox_shape (108419,)
        """

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # print("leftx", leftx)
        # print("len(leftx)", len(leftx))
        # print("lefty", lefty)
        # print("len(lefty)", len(lefty))
        # print("rightx", rightx)
        # print("len(rightx)", len(rightx))
        # print("righty", righty)
        # print("len(righty)", len(righty))

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        # print("left_fit", left_fit)  # 3 adet katsayı verdi => [  3.52432433e-05  -3.39855222e-02   3.43401761e+02]

        right_fit = np.polyfit(righty, rightx, 2)
        # print("right_fit", right_fit)  # 3 adet katsayı verdi => [  4.81021395e-05  -3.09236259e-02   9.50709982e+02]

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # print("left_fitx", left_fitx)
        # print("len(left_fitx)", len(left_fitx))  # her bir y pixeli için bir x değeri bulduk sol çizgi
        # # print("right_fitx", right_fitx)
        # print("len(right_fitx)", len(right_fitx))  # her bir y pixeli için bir x değeri bulduk sağ çizgi

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = (
            (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # print("left_fitx :", left_fitx)

        """
        And you're done! But let's visualize the result here as well:
        """
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)

        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        bottom_distance = right_fitx[719] - left_fitx[719]
        top_distance = right_fitx[0] - left_fitx[0]

        # print("bottom_distance", bottom_distance)
        # print("top_distance", top_distance)

        if left_fitx[719] < 150 or right_fitx[719] > 1130:
            self.detected = False

        # looking for paralellness...
        elif top_distance / bottom_distance > 1.10:
            print("alt üstten küçük")
            # üst tarafta merkezden öteye sapma varsa 25 px left and right ekledik merkeze
            self.detected = False

        elif  top_distance / bottom_distance < 0.85:
            print("0.9 'dan lüçük")
            self.detected = False
        # TODO: rakamlar tutunca bu ikisini birleştir...

        else:
            self.detected = True

        if self.detected:

            self.left_fitx.append(left_fitx)
            self.left_fitx = self.left_fitx[-NUM_ITERATIONS_TO_KEEP:]

            self.right_fitx.append(right_fitx)
            self.right_fitx = self.right_fitx[-NUM_ITERATIONS_TO_KEEP:]
            # self.recent_xfitted.append((self.left_fit, self.right_fit))
            # self.recent_xfitted = self.recent_xfitted[-NUM_ITERATIONS_TO_KEEP:]
            #
            curvature = measure_curvature(ploty, left_fit, right_fit, leftx=left_fitx, rightx=right_fitx)
            self.curvature.append(curvature)  # this is for image pipeline
            # self.radius_of_curvature = curvature

            # self.filled_lane.append(filled_lane)
            self.left_fit.append(left_fit)
            self.left_fit = self.left_fit[-NUM_ITERATIONS_TO_KEEP:]  # keep last N elements

            self.right_fit.append(right_fit)
            self.right_fit = self.right_fit[-NUM_ITERATIONS_TO_KEEP:]  # keep last N elements

            self.ploty = ploty

        return result

    def images_pipeline(self):
            # step 1 : Undistort all test images

            # read the test_images folder by glob api
            images = glob.glob("test_images/*.jpg")

            # read the calibration parameters
            dist_pickle = pickle.load(open("calibration_mtx_dist_pickle.p", "rb"))
            mtx = dist_pickle["mtx"]
            dist = dist_pickle["dist"]

            # undistort test images
            undist_images = undistort_images(mtx, dist, images)

            # step 2: create a binary image
            for image_path in undist_images:
                img = cv2.imread(image_path)
                binary_img = binary_image(img)
                write_to_disk(binary_img, image_path=image_path,
                              write_directory="output_images/01_binary_images/", is_binary=True)

            # step 3: apply perspective transform to all test images
            binary_images = glob.glob("output_images/01_binary_images/*.jpg")
            for image_path in binary_images:
                img = cv2.imread(image_path)
                warped_img = warper(img, src=SRC, dst=DST)
                write_to_disk(warped_img, image_path=image_path,
                              write_directory="output_images/02_perspective_transform//")

            # step 4: Draw lane line pixels been identified in the rectified image and fit with a polynomial
            warped_images = glob.glob("output_images/02_perspective_transform/*.jpg")
            for image_path in warped_images:
                img = cv2.imread(image_path)
                lines_plotted = self.mark_lines(img)
                if self.detected:
                    # print(len(self.left_fitx))
                    # print(len(self.right_fitx))
                    filled = self.fill_lane_area(lines_plotted, self.ploty, self.left_fitx[-1], self.right_fitx[-1])
                else:
                    leftfitx, rightfitx = self.calculate_mean_fit()
                    filled = self.fill_lane_area(lines_plotted, self.ploty, leftfitx, rightfitx)

                self.filled_lane.append(filled)
                write_to_disk(filled, image_path=image_path,
                              write_directory="output_images/03_lane_lines/")

            # step 5: Fill lane line area on uwarped image and write line parameter on image
            for i, image_path in enumerate(undist_images):
                # print(len(self.filled_lane[i]))

                image = cv2.imread(image_path)
                if self.detected:
                    unwarped_lane = warper(self.filled_lane[i], src=SRC, dst=DST, inv=True)

                result = cv2.addWeighted(image, 1, unwarped_lane, 0.3, 0)
                put_curvature = cv2.putText(result, "Radius of Curvature: {}".format(self.curvature[i]), (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                vehicle_position = calculate_vehicle_pos(self.lane_centers[i])
                put_position = cv2.putText(put_curvature, "{}".format(vehicle_position), (10, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                write_to_disk(put_position, image_path=image_path,
                              write_directory="output_images/04_unwarped_images/")


# images_pipeline()
# pipeline = ImgPipeLine()
# pipeline.images_pipeline()


class VideoPipeline(ImgPipeLine):

    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.last_images = []

    def full_pipe_for_single_image(self, img):

        self.last_images.append(img)
        self.last_images = self.last_images[-2:]  # always keep last 2 images (how elegant way of building a stack :) )

        # # mean image
        mean_image = np.array(np.mean(self.last_images, axis=(0)), dtype=np.uint8)

        # read the calibration parameters
        dist_pickle = pickle.load(open("calibration_mtx_dist_pickle.p", "rb"))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

        undistorted = cv2.undistort(mean_image, mtx, dist, None, mtx)
        binary_img = binary_image(undistorted)
        warped_img = warper(binary_img, src=SRC, dst=DST)
        warped_color = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
        lines_plotted = self.mark_lines(warped_color)  # this adds "filled_lane" to "self.filled_lane" array.
        if self.detected:
            # print(self.left_fit)
            # print(self.right_fit)
            filled = self.fill_lane_area(lines_plotted, self.ploty, self.left_fitx[-1], self.right_fitx[-1])
        else:
            leftfitx, rightfitx = self.calculate_mean_fit()
            filled = self.fill_lane_area(lines_plotted, self.ploty, leftfitx, rightfitx)
            # TODO: curvature hesapla!
        # print(filled.shape)
        self.filled_lane.append(filled)
        unwarped_lane = warper(self.filled_lane[self.image_index], src=SRC, dst=DST, inv=True)

        result = cv2.addWeighted(undistorted, 1, unwarped_lane, 0.3, 0)
        put_curvature = cv2.putText(result, "Lines detected: {}".format(self.detected), (10, 50),
                                    cv2.FONT_HERSHEY_PLAIN,  2, (255, 255, 255), 1)

        # put_curvature = cv2.putText(result, "Radius of Curvature: {}".format(self.curvature[self.image_index]), (10, 50),
        #                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        #
        # vehicle_position = calculate_vehicle_pos(self.lane_centers[self.image_index])
        # put_position = cv2.putText(put_curvature, "{}".format(vehicle_position), (10, 110),
        #                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        self.image_index += 1
        # return put_position
        return put_curvature

    def video_pipeline(self, input_video_path, output_video_path):

        clip1 = VideoFileClip(input_video_path)
        final_clip = clip1.fl_image(self.full_pipe_for_single_image)
        final_clip.write_videofile(output_video_path, audio=False)

        return True

#
video_pipe = VideoPipeline()
video_pipe.video_pipeline(input_video_path="project_video.mp4", output_video_path="output_videos/project_video.mp4")
# video_pipe.video_pipeline(input_video_path="challenge_video.mp4", output_video_path="output_videos/challenge_video.mp4")
# video_pipe.video_pipeline(input_video_path="harder_challenge_video.mp4",
#                           output_video_path="output_videos/harder_challenge_video.mp4")





