import cv2
import numpy as np
from matplotlib import pyplot as plt

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img = cv2.imread('Test/Test1.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('Templates/MrRiceTemplate.jpg', 0)
#template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
height, width = template.shape[::]

# detect features from the image
keypoints, descriptors = sift.detectAndCompute(img, None)
keypoints_t, descriptors_t = sift.detectAndCompute(template, None)

# draw the detected key points
sift_image = cv2.drawKeypoints(img_gray, keypoints, img)
sift_template = cv2.drawKeypoints(template, keypoints_t, template)
# show the image
cv2.imshow('image', sift_image)
cv2.imshow('image', sift_template)

# match descriptors of both images
matches = bf.match(descriptors, descriptors_t, 2)

# draw matches
matched_img = cv2.drawMatches(img, keypoints, template, keypoints_t, matches[:], template, flags=2)

# show the image
cv2.imshow('image', matched_img)
cv2.imwrite("matched_images.jpg", matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()