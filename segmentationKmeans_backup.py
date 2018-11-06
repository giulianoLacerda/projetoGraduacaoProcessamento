#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt


class Segment:
    def __init__(self, segments=5):
        # define number of segments, with default 5
        self.segments = segments

    def kmeans(self, image):
        image = cv2.GaussianBlur(image, (7, 7), 0)
        vectorized = image.reshape(-1, 1)
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
        ret, label, center = cv2.kmeans(vectorized, self.segments,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        res = center[label.flatten()]
        segmented_image = res.reshape((image.shape))
        #print len(segmented_image)
        return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

    def extractComponent(self, image, label_image, label, labell):
        component = np.zeros(image.shape, np.uint8)
        component[label_image == label] = image[label_image == label]
        component[label_image == labell] = image[label_image == labell]
        return component


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-n", "--segments", required=False, type=int, help="# of clusters")
    args = vars(ap.parse_args())

    image = misc.imread(args["image"])
    imageRGB = misc.imread("giu1.png")
    r, g, b = cv2.split(imageRGB)
    #print("aqui")
    #print(cv2.countNonZero(g))
    if len(sys.argv) == 3:

        seg = Segment()
        label, result = seg.kmeans(image)
    else:
        seg = Segment(args["segments"])
        label, result = seg.kmeans(image)
    #plt.imshow(result)
    #plt.show()
    result = seg.extractComponent(image, label, 2, 2)
    ret, th = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY)
    thi = np.bitwise_not(th)
    #misc.imsave('aires1_result.jpg', thi)
    plantWd = cv2.bitwise_and(imageRGB, imageRGB, mask=thi)

    rwd, gwd, bwd = cv2.split(plantWd)
    rwd, gwd, bwd = cv2.split(plantWd)
    areaP = cv2.countNonZero(g)
    areaD = areaP - cv2.countNonZero(gwd)
    percent = np.float32(areaD) / np.float32(areaP) * 100
    print(areaP)
    print(areaD)
    print(percent)
    misc.imsave('giu1_result.png',plantWd)



    result = seg.extractComponent(imageRGB, label, 1, 1)
    #misc.imsave('aires1_disease.jpg', result)

    plt.imshow(result)
    plt.show()
    #cv2.waitKey(0)
