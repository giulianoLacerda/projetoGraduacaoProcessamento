from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def pixel_rgb2ycbcr(pixelRGB):
    a = [[65.481/256,128.553/256,24.966/256],
         [-37.797/256,-74.203/256,112/256],
         [112/256,-93.786/256,-18.214/256]]

    a2 = [[0.299, -0.168935, 0.499813],
          [0.587, -0.331665, -0.418531],
          [0.114, 0.50059, -0.081282]]

    a3 = [[65.481, 128.553, 24.966],
          [-37.797, -74.203, 112],
          [112, -93.786, -18.214]]

    a4 = [[0.299, 0.587, 0.114],
          [-0.168736, -0.331264, 0.5],
          [0.5, -0.418688, -0.081312]]

    b = [[0],[128],[128]]

    mtrx_rgb2ycbcr = np.matrix(a2).T
    #print (mtrx_rgb2ycbcr)
    mtrx_rgb = np.matrix(pixelRGB)
    mtrx_b = np.matrix(b)
    return np.uint8(b+(mtrx_rgb2ycbcr*mtrx_rgb.T))


def pixel_rgb2ycgcr(pixelRGB):
    a = [[65.481/256, 128.553/256, 24.966/256],
         [-81.085/256, 112/256, -30.915/256],
         [112/256, -93.786/256, -18.214/256]]

    b = [[16], [128], [128]]

    mtrx_rgb2ycgcr = np.matrix(a)
    print(mtrx_rgb2ycgcr)
    mtrx_rgb = np.matrix(pixelRGB).T
    mtrx_b = np.matrix(b).T
    print(mtrx_rgb2ycgcr*mtrx_rgb)
    return (b+(mtrx_rgb2ycgcr*mtrx_rgb))

def img_rgb2ycbcr(imageRGB):
    print (imageRGB.shape[0])
    imageYCbCr = np.zeros((imageRGB.shape[0], imageRGB.shape[1],3))  # init 2d numpy array
    for row in range(len(imageRGB)):
        for col in range(len(imageRGB[row])):
            #print(pixel_rgb2ycbcr(imageRGB[row][col]))
            #print(imageYCbCr[row][col])
            #print(imageRGB[row][col])
            imageYCbCr[row][col] = (pixel_rgb2ycbcr(imageRGB[row][col]).T)
            #print(imageYCbCr[row][col])
    return np.uint8(imageYCbCr)

def rgb2ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    # Y
    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b
    # Cb
    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b
    # Cr
    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)

# Load Image.
#imageRgb = misc.imread('./Imgs/aires1.jpg')
#imageYcbcr = np.zeros((imageRgb.shape[0], imageRgb.shape[1])) # init 2d numpy array


# Convert image from RGB to YCbCr color space.
#imageLab = np.uint8(color.rgb2ycbcr(imageRgb))
#plt.imshow(imageLab)
#plt.show()



#y,cb,cr = cv2.split(imageLab)


#plt.imshow(y,cmap=cm.Greys_r)
#plt.show()

#plt.imshow(cb,cmap=cm.Greys_r)
#plt.show()

#plt.imshow(cr,cmap=cm.Greys_r)
#plt.show()


#misc.imsave('cafeAires1_y.jpg',y)
#misc.imsave('cafeAires1_cb.jpg',cb)
#misc.imsave('cafeAires1_cr.jpg',cr)

