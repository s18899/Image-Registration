# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:04:30 2020

@author: ADMIN
"""


import cv2 as cv
import numpy as np
import glob 

#--- dimensions of the cropped and aligned card 
j = 1
imgFiles = []
for i in glob.glob("*.jpg"):
    imgFiles.append(i)

img2 = cv.imread('Original\original.jpg')
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
org = cv.imread('Original\original.jpg')
for i in imgFiles:
    img = cv.imread(i,cv.IMREAD_GRAYSCALE)
    img1 = img.copy()
    
    orb = cv.ORB_create(1000)                     #Initiating  ORB points

    kp1, des1 = orb.detectAndCompute(img1, None)  #detecting keypoints and descriptors
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(des1,des2,None)

    matches = sorted(matches, key = lambda x:x.distance)
    
    points1 = np.zeros((len(matches),2), dtype = np.float32)
    points2 = np.zeros((len(matches),2), dtype = np.float32)
    
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 
    #noOfGud = int(len(matches) * GOOD_PERCENT)
    #matches = matches[:noOfGud]    
    
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    
    h, mask = cv.findHomography(points1, points2, cv.RANSAC)
    
#height = img2.shape[0]       #to register an image the reference of the original image is to be used
#width = img2.shape[1]
#channels = img2.shape[2]
    height, width, channels = org.shape

    imgReg = cv.warpPerspective(org, h, (width, height))
    
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:1000], None)

    cv.imshow("Registered Image", imgReg)
    #
    imgReg = cv.cvtColor(imgReg,cv.COLOR_GRAY2RGB)
    filename = 'RegisteredImage%d.jpg' %(j)
    cv.imwrite(filename,imgReg)
    #imgFile = cv.cvtColor(filename)
    cv.imshow("Registered",imgReg)
    cv.imshow("Key Points",img3)

    cv.waitKey(0)
    cv.destroyAllWindows()
    j += 1