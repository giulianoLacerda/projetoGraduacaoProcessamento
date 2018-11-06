#!/usr/bin/python

from scipy import misc
from skimage import color
from skimage.measure import regionprops
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys

def toBinaryPxl(pixel):
    if pixel != (0,0,0):
        return (1,1,1)
    else:
        return (0,0,0)

def toBinaryImg(img):
    binary = np.zeros((imag.shape[0],image.shape[1],image.shape[2]))
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            binary = toBinaryPxl(img[row][col])
    return binary

def calcArea(plantSeg,diseaseSeg):
    areaP=0
    areaD=0
    for row in range(plantSeg.shape[0]):
        for col in range(plantSeg.shape[1]):
            if all(a != 0 for a in plantSeg[row][col]):
                areaP+=1

    area=0
    for row in range(diseaseSeg.shape[0]):
        for col in range(diseaseSeg.shape[1]):
            if all(a != 0 for a in diseaseSeg[row][col]):
                area+=1

    areaD = areaP-area
    print(areaP)
    print(areaD)
    areas=(areaP,areaD)
    return areas

def calcTotal():
    #print("chamou aqui")
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    # print(args["image"])
    # Read image
    imageRgb = misc.imread(args["image"])
    imageBgr = cv2.cvtColor(imageRgb,cv2.COLOR_RGB2BGR)
    frame = cv2.GaussianBlur(imageBgr,(5,5),0)

    # Convert image to HSV from RGB
    imageHsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    # Array for final values
    # Plant
    hsvlowP = np.array([0,0,0])
    hsvhighP = np.array([85,255,255])
    #hsvlowP = np.array([0,126,0])
    #hsvhighP = np.array([179,255,255])

    # Disease
    hsvlowD = np.array([29,0,0])
    hsvhighD = np.array([85,255,255])
    #hsvlowD = np.array([41,126,0])
    #hsvhighD = np.array([179,255,255])

    # Apply the range on a mask
    maskP = cv2.inRange(imageHsv,hsvlowP,hsvhighP)
    maskD = cv2.inRange(imageHsv,hsvlowD,hsvhighD)

    plant = cv2.bitwise_and(frame,frame, mask=maskP)
    plant = cv2.cvtColor(plant, cv2.COLOR_BGR2RGB)
    #maskPP = np.logical_not(maskP)

    #print(plant.shape(0))

    plantWd = cv2.bitwise_and(frame,frame, mask=maskD)
    plantWd = cv2.cvtColor(plantWd, cv2.COLOR_BGR2RGB)

    r,g,b = cv2.split(plant)
    rwd,gwd,bwd = cv2.split(plantWd)

    #areas = calcArea(plant,plantWd)
    areaP = cv2.countNonZero(g)
    areaD = areaP - cv2.countNonZero(gwd)
    percent = np.float32(areaD)/np.float32(areaP)*100
    #print(areaP)
    #print(areaD)
    print(percent)
    #return percent
    #print("percent")

    #print(areas)

    # Find area
    #for row in range(len(plant))

    # Plot images with subplot
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(imageRgb)
    axarr[0,1].imshow(maskP,cmap=cm.Greys_r)
    axarr[1,0].imshow(maskD,cmap=cm.Greys_r)
    axarr[1,1].imshow(plantWd)
    plt.show()

    #ab = regionprops(plant)
    #print(ab.__len__())

calcTotal()
