## Project Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[original_image]: ./camera_cal/calibration3.jpg
[undistorted_image]: ./output_images/00_undistorted_calibration_images/calibration3.jpg
[undistorted_test_image]: ./output_images/00_undistorted_test_images/test1.jpg
[binary_test_image]: ./output_images/01_binary_images/test1.jpg
[undistorted_test_image_with_src_lines]: ./output_images/00_undistorted_test_images_with_src_lines/straight_lines1.jpg
[warped_test_images_with_src_lines]: ./output_images/02_perspective_transform/warped_straight_lines1.jpg
[marked_lines]: ./output_images/02_perspective_marked_lines/test3.jpg
[unwarped_image]: ./output_images/04_unwarped_images/test3.jpg


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for all the steps is contained `project.py`. I have created a class named `ImgPipeline` for Single Images Pipeline and I created another class derived from `ImgPipeline` class named `VideoPipeline`. I have also some global functions as helper functions. For this step I have created a global function  which name is `calibration_pipeline` (lines between 271 and 288). 
 
I start by preparing "object points", in step 1 by calling `calculate_calibration_points` function (lines between 36 and 84) which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. In order to calculate and save these coefficients to disk, I created another function named `calculate_mtx_and_dist_points` (lines between 87 and 105). After that I applied this distortion correction to the all the calibration images using the `cv2.undistort()` function and save all of them to the `.\output_images\00_undistorted_calibration_images` folder. As a sample of this result, here is a calibration image before and after undistortion: 

Before undistortion:
![Original Image][original_image] 

After undistortion:
![Undistorted Image][undistorted_image]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![undistorted test image][undistorted_test_image]
At this step I applied `cv2.undistort()` function to all the test images and saved all the undistorted images to the `/output_images/00_undistorted_test_images/` folder. The code lines for this step is between 723 and 731.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds (gradx and grady) to generate a binary image after several tests. For this process I have created 2 function `binary_image()` and `binary_image2()`. first function is the same that showed in the lessons but it did not performed well. So I created another one `binary_image2()` and used several combinations in order to convert the images binary and finally found a version which performs better. This function is defined the lines between 229 and 237. This function is also uses two helper functions `abs_sobel_thresh()` (lines between 187-204) and `color_treshold()` (lines between 209-226) . Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![binary test image][binary_test_image]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 278 through 310 in the file `project.py`  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points and a binary parameter inv which I used for unwarp.  I chose the hardcode the source and destination points in the following manner:

```python
SRC = np.float32(
    [[(IMG_SIZE[0] / 2) - 55, IMG_SIZE[1] / 2 + 95],
     [((IMG_SIZE[0] / 6) - 40), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 5 / 6) + 75, IMG_SIZE[1]],
     [(IMG_SIZE[0] / 2 + 55), IMG_SIZE[1] / 2 + 95]])

DST = np.float32(
    [[(IMG_SIZE[0] / 4), 0],
     [(IMG_SIZE[0] / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), IMG_SIZE[1]],
     [(IMG_SIZE[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 586, 455      | 320, 0        | 
| 188.333, 720  | 320, 720      |
| 1111.666, 720 | 960, 720      |
| 695, 455      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![undistorted test image with src lines][undistorted_test_image_with_src_lines]
![warped test image with src lines][warped_test_images_with_src_lines]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To detect lane lines I created a function in the `ImgPipeline`class named `mark_lines()` (lines from 471 to 709). This function takes a warped binary image and used `Sliding Windows and Fit a Polynomial` method which has been taught in the lessons. As we saw in the lesson first I take the histogram of the bottom half of the picture in order to detect the lines. After that I applied the sliding windows method through the both left and right lines and after that I applied the polynomial fit method in order to draw the lines.
The applied fit formula is as below:

`f(y) = Ay**2 + By + C`

After applying all of the calculations, here the result for one of my test images with marked with lane lines:

![marked lines][marked_lines]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did these processes in my global functions. For the curvature calculation I created the function named `measure_curvature()` lines 333 through 372 in my code. This function takes 4 arguments in order to calculate the curvature. The argumanets are: ploty, left_fit, right_fit leftx and rightx which all have been calculated in `mark_lines()` function.

For calculating the position of the vehicle, I created `calculate_vehicle_pos()` function (lines 375 through 385). This function takes the center of the lane parameter which has been calculated in `fill_lane_area()` (lines 448 through 469) function.  


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 772 through 791 in my code.  Here is an example of my result on a test image:

![unwarped image][unwarped_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The first approach I took is to apply exact same steps which I learned in the lessons. It worked somehow, but only almost on the half of the video. After that I took the average of the last 2 frames in order to get more stable results. I achieved an improvement but it was very small. After that I decided to change my binary image algorithm. I tried several methods, like detecting the yellow and white colors etc. but none of them worked well enough and finally I succeded with gradx, grady and color_treshold combination. With this algortihm lane lines calculated well in almost 95% of the video. I could not improved the algorithm more than that. So, I decided to use the average of the last 5 line calculations if the algorithm cannot find a line. I also used that method if the detected lines are not parrallel. By doing all of them I finally succeded with all of the video.

But this pipeline is not robust against if the lines cannot be detected more than 1-2 seconds. During all of my works I arrived at the conclusion that there was no any solution which works all of the road conditions. So, if I were going to pursue this project further, I would try to find a way to change the algorithm according to road conditions.