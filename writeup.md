## Writeup

---

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

[image1a]: ./camera_cal/calibration1.jpg "distorted"
[image1b]: ./output_images/calibration1.jpg "Undistorted"
[image2a]: ./output_images/distorted.jpg "Road With distortion"
[image2b]: ./output_images/undistorted.jpg "Road undistorted"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4a]: ./output_images/perspective.jpg "Warp Example"
[image4b]: ./output_images/perspective2.jpg "Warp Example"
[image5a]: ./output_images/color_fit_lines.jpg "Fit Visual Windows"
[image5b]: ./output_images/color_fit_lines2.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./project.ipynb" (or in lines 20 through 55 of the file called `project.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1a]
![alt text][image1b]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2a]
I applied cv2.undistort to this image to get this result:
![alt text][image2b]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 71 through 147 in `project.py`).  

I use a function called combined_thresh to return a binary image. This function combines 5 binary images returned from the functions hls_select, abs_sobel_thresh, mag_thresh, and dir_thresholding. abs_sobel_thresh is called twice in both the y and x directions for gradient. 

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

hls_select transforms the image from RGB color space to HLS color space. Using only the S channel from this color space to create a binary image. Thresholding is used on this just before returning the image.

abs_sobel_thresh, mag_thresh, and dir threshold all convert to grayscale image and apply a sobel operation. They all apply thresholding. Gradiant magnitude is used in thresholding in mag_thresh and direction of the gradiant is used in dir_thresholding.

Here's an example of my output for this step.

![alt text][image3]

Here's a [link to my example (combined_thresh)](./output_images/project_video_combined_thresh.mp4)
, Here's a [link to my example (combined_all)](./output_images/project_video_all_combined.mp4)

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `bird_transform()` and `reverse_transform()`, which appears in lines 515 through 542 in the file `project.py`.  The `bird_transform()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 569, 720      | 
| 1207, 720     | 711, 720      |
| 0, 0          | 0, 0          |
| 1280, 0       | 1280, 0       |


I had edited the destination x coordinates to move towards the middle in order to achieve parallel lanes. I had the bottom corners x points going to IMAGE_W*15/32 and IMAGE_W_16/32. This did achieve parallel lanes but was not happy with the affect to the overall pipeline. I had already finished the pipeline and tailored it to this transformation. I left it as is with the following results:

![alt text][image4b]![alt text][image4a]

Here's a [link to my example (perspective)](./output_images/project_video_bird.mp4)

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In lines 174 through 260 in `find_lane_pixels()` and In lines 333 through 406 in `search_around_poly()`. In `find_lane_pixels()`, window sliding technique will be used. It starts by taking the bottom half of the binary image. These pixels will be summed vertically into a histogram. The first windows will be positioned on the peak values of the histogram. The peak values of the first half and second half of the histogram. 

A variable called minpix mantains the minimum pixels needed to not recenter the window. If more pixels are found, the window will be recentered on the mean position among the pixels found for the window. These pixels will be appended to the list of pixels identified for the lane. These results will be returned to the functioned that called it. `fit_polynomial()` will use these returned values to fit the polynomial. This is how window sliding identifies and fits the positions with a polynomial.

Once a polynomial is found, it will be used along with a margin to detect the lane in the next frames. The Idea being that once a lane is found, it is not expected to change much. Therefore, we do not need to work as hard to find it. In `search_around_poly()`, the area within a margin of the previously found polynomial will be used to find the lane in the current image frame. So a margin of 30 was used, this means 15 pixels to the left and right of the polynomial for both lanes. In `fit_poly()`, these pixels that exist in our margins will be used to fit a polynomial. 

On the left is window sliding, and on the right is finding lanes from prior search:

![alt text][image5a]![alt text][image5b]

Here's a [link to my example (window sliding)](./output_images/project_video_window_sliding.mp4)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented the drawing the information in `draw_info()` on line 570 through 582. This does basic calculations and draws the text onto the image frame. The variables are global, left_curveradius, right_curveradius, and center_offset. In `fit_polynomial()` and `fit_poly()`, at the bottom of the functions, center offset is calculated based on the x pixel closest to the car. Using the two x pixel points and the center location which is half the image width, the offset is calculated.

In `measure_curvature_real()` in lines 486 through 512, the curve radius for both lanes are calculated. Taking the points found for the polynomial, a new polynomial is found. The reason for this is to add the conversion for pixels to meters. After polynomial fitting, the curve is found based on the radius of the curvature. I'm assuming a smaller radius means tighter turns? 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of the final result:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

I tried to implement things in the order which presented incrementally. This worked fairly well. When I did not stop and polish a particular step like perspective transformation, it makes it hard to tweak without it affecting the whole pipeline. This is a good argument for not pushing ahead but sometimes people just can't focus on a particular problem. 

I probably should've designed this pipeline with more dynamic adjustments to allow editing of the pipeline. Particularly the perspective shifting and window combining at the end. I have it hard coded to cut the region I'm interested in and later I add borders back to merge the images. This should've been done dynamically instead of hard coded borders and region cutting. That would've allowed easier manipulation of perspective without needing to tweak the rest of the pipeline.

I had issues displaying binary images, images, etc. I ended up learning that types and format is important, I manipulated the image arrays to display in opencv. From changing array types to uint8, multiplying array by 255 to make it visible. Combining itself with itself 3 times to get 3 channels back. I managed to get everything displayed in openCv. I also managed to convert it back to binary by taking a single channel and dividing by 255. There was a few times I ended up needing to define the array type due to my perhaps spaghetti manipulation of the images. I wanted to get video and images in opencv. Image manipulation was an issue for me, but I learned a lot from it. 

I'll consider two types of failure. I think that it will probably fail/crash if a polynomial is not found during some polynomial fitting calls. It is not designed robust enough to continue at all parts, I designed it just for the project video, and it assumes no issues at times. In a car, this is bad. One bad image frame could crash it. 

Logical errors is the other type. There might be some in curve measurement but that's because I'm not too familiar with what it should be. I roughly edited the per pixels to try and get something that seems right. I did take into account how many pixels were in my image slice. 

I think designing a strategy for bad frames is one. Does one use previous frame and guess? use previous frame data to estimate current lane positions? 

I should improve the perspective transformation. I could also add adaptive thresholding. There are tweaks that could improve it. I think adding some more channels might help. I think improving the combined binary image and perspective are my main areas I'd like to improve. But there are also areas in techniques for dealing with bad frames and data that I could do to improve it. If only one lane were detected, one could guess where the other lane is based on road width. Techniques for dealing with bad frames.