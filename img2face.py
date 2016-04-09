"""Open image, find face, clip it and save the face as new image."""

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('sample.JPG',0)