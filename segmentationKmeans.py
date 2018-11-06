#!/usr/bin/env python

import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt

def kmeans(clusters, image):
    image = cv2.GaussianBlur(image, (7, 7), 0)
    vectorized = image.reshape(-1, 1)
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    ret, label, center = cv2.kmeans(vectorized, clusters,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    #print len(segmented_image)
    return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

def extractComponent(clusters, image, label_image, label, labell):
    component = np.zeros(image.shape, np.uint8)
    component[label_image == label] = image[label_image == label]
    component[label_image == labell] = image[label_image == labell]
    return component

def method_kmeans(imageCb):
    imageRGB = misc.imread("giu1.png")
    r, g, b = cv2.split(imageRGB)
    label, result = seg.kmeans(3,imageCb)
    result = extractComponent(image, label, 2, 2)
    ret, th = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)
    thi = np.bitwise_not(th)
    #misc.imsave('aires1_result.jpg', thi)
    plantWd = cv2.bitwise_and(imageRGB, imageRGB, mask=thi)
    plt.imshow(plantWd)
    plt.show()
