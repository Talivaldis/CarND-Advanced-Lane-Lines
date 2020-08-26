# importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

global left_fit, right_fit, left_curveradius, right_curveradius, mtx, dist, center_offset
left_fit = None
right_fit = None
left_curveradius = 9999
right_curveradius = 9999
center_offset = 9999


def calibrate_camera():
    global mtx, dist
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('./camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        nx = 9

        # If found, add object points, image points
        if ret == True:
            src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    cwd = os.getcwd()
    print(cwd)

    for fname in images:
        img = cv2.imread(fname)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        img = cv2.undistort(img, mtx, dist, None, mtx)
        temp = os.path.split(fname)
        fname = "./output_images/" + temp[1]
        print(fname)
        # cv2.imwrite(fname, img)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

    cv2.destroyAllWindows()


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the binary image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def combined_thresh(image):
    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
    grady = abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi / 2))
    s_binary = hls_select(image, thresh=(90, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1

    return combined


def video_process():
    calibrate_camera()
    video_output = 'output_images/project_video.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("project_video.mp4")
    clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    clip.write_videofile(video_output, audio=False)
    return


def process_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.undistort(img, mtx, dist, None, mtx)
    img = pipeline(img)
    cv2.imshow('img', img)
    cv2.waitKey(30)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def find_lane_pixels(binary_warped):
    out_img = binary_warped.copy()
    binary_warped = binary_warped[:, :, 1]
    binary_warped = binary_warped / 255

    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

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
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    global left_fit, right_fit, center_offset
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        # left_curveradius, right_curveradius = measure_curvature_real(ploty, leftx, rightx)
        measure_curvature_real(lefty, righty, leftx, rightx)
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    out_img = my_draw(out_img, left_fitx, ploty)
    out_img = my_draw(out_img, right_fitx, ploty)
    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    right_start = rightx[len(rightx) - 1]
    left_start = leftx[len(leftx) - 1]
    difference = right_start - left_start
    center_of_lane = left_start + difference / 2
    center_offset = out_img.shape[1] / 2 - center_of_lane
    # 3.7 / 220
    # center_offset = center_offset * (3.7 / 220)
    center_offset = center_offset * (3.7 / 700)

    return out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    global left_fit, right_fit, center_offset
    ###Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # left_curverad, right_curverad = measure_curvature_real(ploty, leftx, rightx)
    measure_curvature_real(lefty, righty, leftx, rightx)

    right_start = rightx[len(rightx) - 1]
    left_start = leftx[len(leftx) - 1]
    difference = right_start - left_start
    center_of_lane = left_start + difference / 2
    center_offset = img_shape[1] / 2 - center_of_lane
    # 3.7 / 220
    # center_offset = center_offset*(3.7/220)
    center_offset = center_offset * (3.7 / 700)

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped):
    binary_warped = binary_warped[:, :, 1]
    binary_warped = binary_warped / 255
    global left_fit
    global right_fit
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 30

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    # Recast the x and y points into usable format for cv2.fillPoly()
    # line_pts = np.hstack((left_fitx, right_fitx))
    line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                                                               ploty])))], dtype=np.int32)
    line_pts = np.hstack((line_window1, line_window2))
    cv2.fillPoly(window_img, line_pts, (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    result = my_draw(result, left_fitx, ploty)
    result = my_draw(result, right_fitx, ploty)
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result


def window(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return


def pipeline(img):
    original = img.copy()
    img = combined_thresh(img)
    img = np.dstack((img, img, img)) * 255
    img = np.asarray(img, np.uint8)

    height = img.shape[0]
    width = img.shape[1]
    polygons = np.array([[(0 + 40, height - 50), ((width / 2) - 80, 5 * height / 8), ((width / 2) + 80, 5 * height / 8),
                          (width - 40, height - 50)]], np.int32)
    img = region_of_interest(img, polygons)

    img = bird_transform(img)

    global left_fit, right_fit

    if left_fit is None and right_fit is None:
        img = fit_polynomial(img)
    else:
        img = search_around_poly(img)

    img = reverse_transform(img, original.shape[:2])
    img = np.asarray(img, np.uint8)
    img = cv2.addWeighted(img, 0.7, original, 1, 0)

    img = draw_info(img)

    return img


def measure_curvature_real(lefty, righty, leftx, rightx):
    global left_curveradius
    global right_curveradius
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 220  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(lefty) + 1

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    left_curveradius = left_curverad
    right_curveradius = right_curverad

    return left_curverad, right_curverad


def bird_transform(image):
    image_shape = image.shape[:2]
    IMAGE_H = image_shape[0]
    IMAGE_W = image_shape[1]
    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    image = image[450:(450 + IMAGE_H), 0:IMAGE_W]
    image = cv2.warpPerspective(image, M, (IMAGE_W, IMAGE_H))
    width = image.shape[1]
    height = image.shape[0]
    image = image[0:height - 160, 400:width - 400]
    return image


def reverse_transform(image, shape):
    image_shape = shape
    IMAGE_H = image_shape[0]
    IMAGE_W = image_shape[1]
    value = (0, 0, 0)
    image = cv2.copyMakeBorder(image, 0, 160, 400, 400, cv2.BORDER_CONSTANT, None, value)
    src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    dst = np.float32([[569, IMAGE_H], [711, IMAGE_H], [0, 0], [IMAGE_W, 0]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    image = cv2.warpPerspective(image, Minv, (IMAGE_W, IMAGE_H))
    image = image[0:(IMAGE_H - 1170), 0:IMAGE_W]
    image = cv2.copyMakeBorder(image, 450, 0, 0, 0, cv2.BORDER_CONSTANT, None, value)
    return image


def region_of_interest(image, polygons):
    mask = np.zeros_like(image)
    match_mask_color = (255, 255, 255)
    cv2.fillPoly(mask, polygons, match_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def draw_roi(image, polygons):
    color = (0, 0, 255)
    thickness = 5
    for index in range(len(polygons[0]) - 1):
        image = cv2.line(image, tuple(polygons[0][index]), tuple(polygons[0][index + 1]), color, thickness)
    image = cv2.line(image, tuple(polygons[0][0]), tuple(polygons[0][len(polygons[0]) - 1]), color, thickness)
    return image


def my_draw(img, fit_x, ploty):
    color = (255, 255, 0)
    thickness = 2
    for i in range(len(fit_x) - 1):
        img = cv2.line(img, (int(fit_x[i]), int(ploty[i])), (int(fit_x[i + 1]), int(ploty[i + 1])), color, thickness)
    return img


def draw_info(image):
    average_radius = (left_curveradius + right_curveradius) / 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2
    text = f"curve radius[m]: {average_radius:.3f}"
    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    text = f"center offset[m]: {center_offset:.4f}"
    org = (50, 100)
    image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


def main():
    print("running test")
    video_process()


main()
