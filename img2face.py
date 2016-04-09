"""Open image, find face, clip it and save the face as new image."""
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


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

# image_path = sys.argv[1]
# casc_path = sys.argv[2]
#
# # Read the image
# image = cv2.imread(image_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# face_cascade = cv2.CascadeClassifier(casc_path)
#
# faces = face_cascade.detectMultiScale(
#     gray,
#     scaleFactor=1.2,
#     minNeighbors=5,
#     minSize=(10, 10),
#     flags=cv2.cv.CV_HAAR_SCALE_IMAGE
# )
#
# print "Found {0} faces!".format(len(faces))
#
# # Draw a rectangle around the faces
# i = 3
# for (x, y, w, h) in faces:
#     i += 1
#     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     crop_img = image[y:y+h, x:x+h]
#     cv2.imwrite('faces/face{}.png'.format(i), crop_img)
#
# cv2.imshow("Faces found", image)
# key = cv2.waitKey(0)
# if key == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
#
# # filename = 'sample.JPG'
# #
# # img = cv2.imread(filename, 0)
# # plt.imshow(img, cmap='gray', interpolation='bicubic')
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# # plt.show()
# #
# #
# # # Load an color image in grayscale
# #
# # img = cv2.imread(filename, 0)
# # w = "mywindow"
# # cv2.namedWindow(w, cv2.CV_WINDOW_AUTOSIZE)
# # cv2.resizeWindow(w, 128, 128)
# #
# # #cv2.imwrite('face.png', img)
# #
# # cv2.imshow(w, img)
# # cv2.resizeWindow(w, 128, 128)
# # key = cv2.waitKey(0)
# # if key == 27:         # wait for ESC key to exit
# #     cv2.destroyAllWindows()