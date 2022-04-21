import cv2
import numpy as np
# from matplotlib import pyplot as plt
# import imutils
import os

# https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/

# blood cell detection
def cell_detect(image):
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
  circles = np.concatenate((circles, [[[0]]*circles.shape[1]]), axis = 2)
  distances = np.linalg.norm(circles[0, :, :2] - circles[0, :, :2][:, np.newaxis], axis = -1)
  
  # identify clumps
  threshold = 15
  for i in range(distances.shape[0]):
    close = 0
    for j in range(distances.shape[1]):
      if(distances[i, j] == 0):
        continue
      if(distances[i, j] <= threshold):
        close += 1
    if(close >= 3):
      circles[0, i, -1] = 1
  for i in range(distances.shape[0]):
    close = 0
    for j in range(distances.shape[1]):
      if(distances[i, j] == 0):
        continue
      if(distances[i, j] <= threshold and circles[0, j, -1] == 1):
        close += 1
    if(close >= 1):
      circles[0, i, -1] = 1
    else:
      circles[0, i, -1] = 0
  
  # draw circles -- circles in clumps are red, otherwise green
  circles = np.int_(np.round(circles[0, :]))
  for x, y, r, clumped in circles:
    color = (0, 0, 255) if(clumped) else (0, 255, 0)
    cv2.circle(blood_cells, (x, y), r, color, 1)
  
  # update image with detected circles
  orig_image[:, left_tube_wall:right_tube_wall] = blood_cells
  return orig_image

# input mp4 file
vidcap = cv2.VideoCapture("dextran_v01.mp4")
success, img = vidcap.read()

# detect tube inner wall indices
# tube width normalized to 72
img_brt = np.mean(img, axis = (0, 2))
img_brt = np.where(img_brt <= 88)[0]
wall_ind = np.where(img_brt[1:] - img_brt[:-1] > 50)[0][1]
right_tube_wall = (img_brt[wall_ind] + img_brt[wall_ind + 1])//2 + 37
left_tube_wall = right_tube_wall - 72

# begin cell detection
img = cell_detect(img)

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
  img = cell_detect(img)
  if(count > 10):
    quit()

# plt.imshow(img, cmap = "gray")
# plt.title("Frame 10 Circles"), plt.xticks([]), plt.yticks([])
# plt.show()

frames = count

# create video
name = "dextran_v02c_detect_clumps.mp4"
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
