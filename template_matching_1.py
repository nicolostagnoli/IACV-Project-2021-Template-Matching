import cv2 as cv
import numpy as np
import argparse
import numpy.ma as ma
from customRansac import customFindHomography
from customRansac import customFindHomography3D
import time

img_scene = cv.imread("3D/1/rgb_image.jpg")
img_object = cv.imread("Templates3/barchette_intera.jpg")
point_cloud = np.load("3D/1/pointCloud.npy") #[height][width][xyz]

#Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 1
detector = cv.xfeatures2d_SIFT.create()
keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

#Matching descriptor vectors with a FLANN based matcher
matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

#Filter matches using the Lowe's ratio test
ratio_thresh = 0.9
good_matches = []
for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

#Draw matches
img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

oldSize = len(good_matches)
newSize = -1
j = 0

start = time.time()
###Sequential RANSAC
while((oldSize != newSize) and len(good_matches) >= 4):
    oldSize = len(good_matches)

    #Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    
    #cv.findHomography(obj, scene, cv.RANSAC, confidence = 0.995, ransacReprojThreshold=5)
    H, mask =  customFindHomography(obj, scene, 0.55)
    #H, mask =  customFindHomography3D(obj, scene, point_cloud, 0.55)
    # H homography from template to scene
    H = np.asarray(H)
    #Take points from the scene that fits with the homography
    img_instance_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    maskk = (mask[:]==[0])
    instance_good_matches = ma.masked_array(good_matches, mask=maskk).compressed()

    #Remove inliers from good matches array
    new_good_matches = np.asarray(list(set(good_matches)-set(instance_good_matches)))
    good_matches = new_good_matches
    newSize = len(good_matches)

    ### Show matches fitting the found homography
    #cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, instance_good_matches, img_instance_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('Good Matches - 1 instance', img_instance_matches)
    #cv.imwrite("output_mask" + str(j) + ".jpg", img_instance_matches)
    #cv.waitKey()

    #Get the corners from the template
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]

    try:
        #Check for degenerate homography
        valid = True
        for k in range(0, len(obj_corners)):
            x = obj_corners[k,0,0]
            y = obj_corners[k,0,1]
            if (H[2][0]*x + H[2][1]*y + H[2][2]) / np.linalg.det(H) <= 0:
                valid = False

        #Draw lines between the corners
        scene_corners = cv.perspectiveTransform(obj_corners, H)

        #Draw Bounding box on the image
        if valid:
            print("-------------------------------DRAWING BOX----------------------------------------")
            cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
                (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
            cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
                (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
            cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
                (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
            cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
                (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    except:
        print("Cannot draw bounding box")
        print(H)
    
    j += 1

 ### End Sequential RANSAC

end = time.time()
print("time elapsed:")
print(end - start)

#Show detected matches
cv.imshow('Good Matches', img_matches)
cv.imwrite("output.jpg", img_matches)