import cv2
import numpy as np
import getopt
import sys
import random



#
# Computers a homography from 4-correspondences
#
def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h

#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    if (estimatep2.item(2) != 0):
        estimatep2 = (1/estimatep2.item(2))*estimatep2
    else:
        return 1000
    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

#
#Runs through ransac algorithm, creating homographies from random correspondences
#
def customFindHomography(obj,scene, thresh):
    
    correspondenceList = []

    for z in range(len(scene[:,0])):
            (x1, y1) = obj[z,0] , obj[z,1]
            (x2, y2) = scene[z,0] , scene[z,1]
            correspondenceList.append([x1, y1, x2, y2])

    corr = np.matrix(correspondenceList)

    maxInliers = []
    finalH = None
    finalMask = np.zeros(shape = (len(obj[:,0])) )
    for i in range(600):

        mask = np.zeros(shape = (len(obj[:,0])) )

        #find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []
        

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 3:
                inliers.append(corr[i])
                mask[i] = 1
                

        if len(inliers) > len(maxInliers):
            
            maxInliers = inliers
            finalH = h
            finalMask = mask

        print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break

    return finalH, finalMask;

