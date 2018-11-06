from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def pixel_rgb2ycgcr(pixelRGB):
    a = np.asarray([[65.481/256, 128.553/256, 24.966/256],
         [-81.085/256, 112/256, -30.915/256],
         [112/256, -93.786/256, -18.214/256]])

    b = np.asarray([[16], [128], [128]])

    mtrx_rgb2ycgcr = np.matrix(a)
    #print(mtrx_rgb2ycgcr)
    mtrx_rgb = np.matrix(pixelRGB).T
    mtrx_b = np.matrix(b).T
    #print(mtrx_rgb2ycgcr*mtrx_rgb)
    return (b+(mtrx_rgb2ycgcr*mtrx_rgb))

def rgb2ycgcr(imageRGB):
    #print (imageRGB.shape[0])
    imageYCgCr = np.zeros((imageRGB.shape[0], imageRGB.shape[1],3))  # init 2d numpy array
    for row in xrange(len(imageRGB)):
        for col in xrange(len(imageRGB[row])):
            #print(pixel_rgb2ycbcr(imageRGB[row][col]))
            #print(imageYCbCr[row][col])
            #print(imageRGB[row][col])
            imageYCgCr[row][col] = (pixel_rgb2ycgcr(imageRGB[row][col]).T)
            #print(imageYCbCr[row][col])
    return imageYCgCr

def rgb2ycgcr_fast(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    a = .25578 * r
    #print(a)
    # Y
    cbcr[:,:,0] = 16 + .25578 * np.asarray(r,dtype='f') + .50216 * np.asarray(g,dtype='f') + .09752 * np.asarray(b,dtype='f')
    # Cb
    cbcr[:,:,1] = 128 - .31673 * np.asarray(r,dtype='f') + .4375 * np.asarray(g,dtype='f') - .12076 * np.asarray(b,dtype='f')
    # Cr
    cbcr[:,:,2] = 128 + .4375 * np.asarray(r,dtype='f') - .36635 * np.asarray(g,dtype='f') - .07114 * np.asarray(b,dtype='f')
    return np.uint8(cbcr)

