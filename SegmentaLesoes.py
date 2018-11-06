#!/usr/bin/env python

from scipy import misc
from skimage import color
from skimage.measure import regionprops
from skimage.measure import find_contours
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import conversion

# Background BLACK
def watershed(imageRgb,imageBinary,plantBinary):

    #plt.imshow(imageBinary,cmap=cm.Greys_r)
    #plt.show()
    #print("inicio watershed")
    imageBinary = np.uint8(imageBinary)
    imageBinary = cv2.bitwise_and(plantBinary, plantBinary, mask=np.uint8(imageBinary))
    imageRgb2 = cv2.bitwise_and(imageRgb, imageRgb, mask=np.uint8(imageBinary))

    # Define kernel
    ####kernel = np.ones((3, 3), np.uint8)
    #opening = cv2.morphologyEx(imageBinary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Area que com certeza e fundo
    ####sure_bg = cv2.dilate(imageBinary, kernel, iterations=5)

    # Area que com certeza nao eh fundo
    ####dist_transform = cv2.distanceTransform(imageBinary, cv2.DIST_L1, 5)
    ####ret, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)

    # Encontra regiao desconhecida
    ####sure_fg = np.uint8(sure_fg)
    ####unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(np.uint8(imageBinary))

    # Add one to all labels so that sure background is not 0, but 1
    markers[1] = markers[1] + 500

    # Now, mark the region of unknown with zero
    ####markers[unknown == 255] = 0
    #plt.imshow(markers)
    #plt.show()

    #markers = cv2.watershed(imageRgb2, markers)


    lesoes = []
    print("Lesoes")
    for indice in xrange(1,ret):

        # Cria mascara com uma unica lesao
        aux = np.zeros((markers.shape[0], markers.shape[1]))
        aux[markers == indice] = 255
        #aux[markers == -1] = 255

        # Aplica essa mascara na imagem original
        lesaoRgb = cv2.bitwise_and(imageRgb2, imageRgb2, mask=np.uint8(aux))

        # Identifica as propriedades dessa unica lesao
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(aux), connectivity=4)

        # Cria uma mascara com as dimensoes retangulares da lesao identificada
        lesaoAux = np.zeros((stats[1][3], stats[1][2], 3), dtype=np.uint8)

        # Preenche a mascara com as dimensoes retangulares da lesao com a lesao em rgb.
        # altura = y0+deltay, comprimento = x0+deltax
        lesaoAux = lesaoRgb[stats[1][1]:stats[1][1] + stats[1][3], stats[1][0]:stats[1][0] + stats[1][2]]

        #plt.imshow(lesaoAux,cmap=cm.Greys_r)
        #plt.show()
        if(np.count_nonzero(lesaoAux)>500):
            lesoes.append(lesaoAux)

    #markers[markers == -1] = 100
    #markers[markers != 100] = 0
    #markers = np.logical_not(markers)
    #nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(markers), connectivity=4)
    return lesoes


def method_ycgcr(imageRgb,plantBinary):
    frameRgb = cv2.GaussianBlur(imageRgb,(5,5),0)
    frameYcgcr = conversion.rgb2ycgcr_fast(frameRgb)
    y,cg,cr = cv2.split(frameYcgcr)

    # Calc from im_dif(x,y) = Cr(x,y)-Cg(x,y)
    im_dif = np.zeros((frameRgb.shape[0], frameRgb.shape[1],1))
    im_dif = np.subtract(cr,cg)

    # Select initial threshold value, D1 and D2
    t = 0
    d1 = np.zeros((frameRgb.shape[0], frameRgb.shape[1]))
    d2 = np.zeros((frameRgb.shape[0], frameRgb.shape[1]))

    # Averages
    m1 = 0
    m2 = 0
    total = im_dif.shape[0]*im_dif.shape[1]

    for a in xrange(0,20):
        print(t)
        condition1 = im_dif > t
        condition2 = im_dif <= t
        d1 = cv2.bitwise_and(im_dif, im_dif, mask=np.uint8(condition1))
        d2 = cv2.bitwise_and(im_dif, im_dif, mask=np.uint8(condition2))
        m1 = np.mean(d1)
        m2 = np.mean(d2)
        t = t = (m1+m2)/2

    #ret,th = cv2.threshold(d1,200,255,cv2.THRESH_BINARY)
    ret1, th1 = cv2.threshold(d2, 0, 255, cv2.THRESH_BINARY_INV)
    plantWd = cv2.bitwise_and(imageRgb, imageRgb, mask=th1)
    #plt.imshow(plantWd)
    #plt.show()
    return plantWd,watershed(imageRgb,thi,np.uint8(plantBinary))

def method_ycgcr2(imageRgb,plantBinary):
    frameGray = cv2.cvtColor(imageRgb, cv2.COLOR_RGB2GRAY)
    frameRgb = cv2.GaussianBlur(imageRgb,(5,5),0)
    frameYcgcr = conversion.rgb2ycgcr_fast(frameRgb)
    y,cg,cr = cv2.split(frameYcgcr)

    # Calc from im_dif(x,y) = Cr(x,y)-Cg(x,y)
    im_dif = np.zeros((frameRgb.shape[0], frameRgb.shape[1],1))
    im_dif = np.subtract(cr,cg)
    #plt.imshow(im_dif,cmap=cm.Greys_r)
    #plt.show()

    # Select initial threshold value, D1 and D2
    t = 0
    d1 = np.zeros((frameRgb.shape[0], frameRgb.shape[1]))
    d2 = np.zeros((frameRgb.shape[0], frameRgb.shape[1]))

    # Averages
    m1 = 0
    m2 = 0
    total = im_dif.shape[0]*im_dif.shape[1]

    dif = 1;
    #print("Aqui")
    while (dif>0.0001):
        d1 = np.where(im_dif >  t,im_dif,0)
        d2 = np.where(im_dif <= t,im_dif,0)
        m1 = np.mean(d1)
        m2 = np.mean(d2)
        t2 = (m1+m2)/2
        dif = abs(t2-t)
        t=t2

    #plt.imshow(d1,cmap=cm.Greys_r)
    #plt.show()
    ret, th = cv2.threshold(d1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thi = np.logical_not(th)

    #mask = np.asarray(th, dtype=np.uint8)
    #mask = mask.astype(int)
    #misc.imsave("136_pretoYCbCr.png",mask)


    plantWd = cv2.bitwise_and(imageRgb, imageRgb, mask=np.uint8(th))
    #plt.imshow(plantWd)
    #plt.show()
    print("Fim threshold")
    return plantWd,watershed(imageRgb,thi,np.uint8(plantBinary))

def method_thresholdHsv(imageRgb):
    plantHsv = cv2.cvtColor(imageRgb, cv2.COLOR_RGB2HSV)
    hsvlowD = np.array([30, 0, 0])
    hsvhighD = np.array([179, 255, 255])
    maskD = cv2.inRange(plantHsv, hsvlowD, hsvhighD)
    kernel1 = (2, 2)
    kernel1 = np.ones(kernel1, np.uint8)
    maskD = cv2.dilate(np.uint8(maskD), kernel1, iterations=11)
    maskD = cv2.erode(np.uint8(maskD), kernel1, iterations=5)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(maskD))
    sizes = stats[:, -1]
    max_label = 1
    max_size = sizes[1]
    for i in xrange(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    maskD = np.zeros(output.shape)
    maskD[output == max_label] = 255
    #maskD = np.logical_not(maskD)

    plantWd = cv2.bitwise_and(imageRgb, imageRgb, mask=np.uint8(maskD))
    #plt.imshow(plantWd,cmap=cm.Greys_r)
    #plt.show()
    return plantWd

def kmeans(clusters, image):
    image = cv2.GaussianBlur(image, (7, 7), 0)
    vectorized = image.reshape(-1, 1)
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    ret, label, center = cv2.kmeans(vectorized, clusters,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return label.reshape((image.shape[0], image.shape[1])), segmented_image.astype(np.uint8)

def extractComponent(image, label_image, label, labell):
    component = np.zeros(image.shape, np.uint8)
    component[label_image == label] = image[label_image == label]
    component[label_image == labell] = image[label_image == labell]
    return component

def method_kmeans(imageRGB,mascara):
    imageYcbcr = np.uint8(color.rgb2ycbcr(imageRGB))
    y, imageCb, imageCr = cv2.split(imageYcbcr)
    label, result = kmeans(3,imageCr)
    result = extractComponent(result, label, 2, 2)
    ret, th = cv2.threshold(result, 0,1, cv2.THRESH_BINARY_INV)
    plantWd = cv2.bitwise_and(imageRGB, imageRGB, mask=th)

    #mask = cv2.bitwise_and(np.uint8(mascara), np.uint8(mascara), mask=th)
    #mask = np.asarray(mask, dtype=np.uint8)
    #mask = mask.astype(int)
    #misc.imsave("136_pretoKmeans.png", mask)

    return plantWd, th


