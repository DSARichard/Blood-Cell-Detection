import cv2
import numpy as np
# from matplotlib import pyplot as plt
import imutils
import os

highlight_flow = False

# https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

# detect optical flow
def flow_detect(prev_image, next_image, output_image = False, output_contours = False):
  # calculate dense optical flow
  flow = cv2.calcOpticalFlowFarneback(prev_image, next_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  
  # convert to color image if specified
  if(output_image):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(img[:, left_tube_wall:right_tube_wall])
    hsv[..., 1] = 255
    hsv[..., 0] = ang*90/np.pi
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_image
  
  # draw detected clumps based on flow if specified
  if(output_contours):
    mag = cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]
    flow_image = np.zeros_like(img[:, left_tube_wall:right_tube_wall])
    values = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # find potential clumps
    threshold = 60
    values[values < threshold] = 0
    values[values >= threshold] = 255
    kernel = np.ones((5, 5), np.uint8)
    values = cv2.morphologyEx(values, cv2.MORPH_CLOSE, kernel)
    values = cv2.erode(values, kernel, iterations = 2)
    values = values.astype(np.uint8)
    
    # detect contours of clumps
    # isolate those with area >= 300
    cnts = cv2.findContours(values.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    clumps = []
    for c in cnts:
      if(cv2.contourArea(c) >= 300):
        clumps.append(c)
    clumps = np.array(clumps, object)
    
    # draw clumps
    flow_image = cv2.drawContours(flow_image.copy(), clumps, -1, (0, 90, 255), -1)
    return flow_image
  
  # otherwise return detect flow
  return flow

# blood cell detection
def cell_detect(image, flow = None, flow_clumps = None):
  # manipulate image
  orig_image = image.copy()
  sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
  image = cv2.filter2D(image[:, left_tube_wall:right_tube_wall], -1, sharpening_kernel)
  blood_cells = np.uint8(image.copy())
  gray_image = np.uint8(cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY))
  
  # detect circles
  circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1.05, 5, param2 = 8.5, minRadius = 1, maxRadius = 7)
  if(circles is None):
    return orig_image
  circles = np.concatenate((circles, np.zeros((1, circles.shape[1], 1), int)), axis = -1)
  distances = np.linalg.norm(circles[0, :, :2] - circles[0, :, :2][:, np.newaxis], axis = -1)
  
  # determine optical flow for centers of detected circles
  if(flow is not None):
    circle_centers = np.int_(np.round(circles[0, :, :2])).T
    circle_flow = flow[np.clip(circle_centers[1], 0, 719), np.clip(circle_centers[0], 0, 71)]
  else:
    circle_flow = np.zeros((circles.shape[1], 2))
  
  # identify circles in clumps: in a continuum from 0 to 511
  # evaluation based on circle–circle distance and alignment of flow direction
  threshold = 15
  for i in range(distances.shape[0]):
    close = 0
    for j in range(distances.shape[1]):
      if(0 < distances[i, j] <= threshold):
        flow_dist = np.linalg.norm(circle_flow[i] - circle_flow[j])
        close += 3*(1 - np.tanh(flow_dist))
    circles[0, i, -1] = np.clip(close, 0, 7)*73
  
  # evaluation based on previous evaluations of nearby circles
  for i in range(distances.shape[0]):
    close = 0
    for j in range(distances.shape[1]):
      if(0 < distances[i, j] <= threshold):
        close += circles[0, j, -1]
    circles[0, i, -1] = np.clip(close, 0, 657)*7/9
  
  # draw circles — circles in clumps are red, green if not, yellow if uncertain
  circles = np.int_(np.round(circles[0, :]))
  for x, y, r, clumped in circles:
    color = (0, int(min(511 - clumped, 255)), int(min(clumped, 255)))
    cv2.circle(blood_cells, (x, y), r, color, -1)
  
  # overlay clumps from flow detection onto image
  if(flow_clumps is not None):
    blood_cells = cv2.addWeighted(blood_cells, 1, flow_clumps, 0.6, 0)
  
  # update image with detected circles
  orig_image[:, left_tube_wall:right_tube_wall] = blood_cells
  return orig_image

# input mp4 file
vidcap = cv2.VideoCapture("dextran_v01.mp4")
success, img = vidcap.read()

# detect tube inner wall indices based on brightness
# tube width normalized to 72
img_brt = np.mean(img, axis = (0, 2))
img_brt = np.where(img_brt <= 88)[0]
wall_ind = np.where(img_brt[1:] - img_brt[:-1] > 50)[0][1]
right_tube_wall = (img_brt[wall_ind] + img_brt[wall_ind + 1])//2 + 37
left_tube_wall = right_tube_wall - 72

# begin cell detection
orig_img = img.copy()
img = cell_detect(img)
prev_img = cv2.cvtColor(np.float32(img[:, left_tube_wall:right_tube_wall]), cv2.COLOR_BGR2GRAY)

# create frames
count = 0
while(success):
  # output current frame with blood cell detection
  cv2.imwrite("frame" + str(count).zfill(3) + ".jpg", img)
  
  # next frame
  success, img = vidcap.read()
  count += 1
  if(img is None):
    continue
  
  # detect flow between current and previous frame with optical flow detection
  next_img = cv2.cvtColor(np.float32(img[:, left_tube_wall:right_tube_wall]), cv2.COLOR_BGR2GRAY)
  flow = flow_detect(prev_img, next_img)
  flow_img = (
    flow_detect(prev_img, next_img, output_contours = True) if(highlight_flow)
    else None
  )
  prev_img = next_img
  
  # use flow of frame 1 on frame 0
  if(count == 1):
    orig_img = cell_detect(orig_img, flow, flow_img)
    cv2.imwrite("frame000.jpg", orig_img)
  
  # move on to next frame
  img = cell_detect(img, flow, flow_img)
  if(count > 10):
    quit()

# plt.imshow(img, cmap = "gray")
# plt.title("Frame"), plt.xticks([]), plt.yticks([])
# plt.show()

frames = count

# create video
version = "02e" if(highlight_flow) else "02d"
name = f"dextran_v{version}_detect_clumps.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(name, fourcc, 60.0, (1280, 720))

# write frames
for count in range(frames):
  # add frame to video
  img = cv2.imread("frame" + str(count).zfill(3) + ".jpg")
  out.write(img)
  
  # remove frame file
  try:
    os.remove("frame" + str(count).zfill(3) + ".jpg")
  # any exceptions
  except:
    pass

out.release()
