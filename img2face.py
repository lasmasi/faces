"""Open image, find face, clip it and save the face as new image."""

import numpy as np
import cv2
import sys

# Load an color image in grayscale
img = cv2.imread('Momo.jpg', 1)
cv2.namedWindow('Momo.jpg', cv2.WINDOW_NORMAL)
#cv2.imshow('Momo.jpg', img)
r = 1500.0 / img.shape[1]
dim = (1500, int(img.shape[0] * r))

# perform the actual resizing of the image and show it
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)




cv2.waitKey(0)
cv2.destroyAllWindows()