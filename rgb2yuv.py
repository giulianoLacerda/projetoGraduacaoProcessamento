from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


imageRgb = misc.imread("./Imgs/aires1.jpg")
imageYuv = cv2.cvtColor(imageRgb,cv2.COLOR_RGB2YUV)
y,u,v = cv2.split(imageYuv)
imageYcbcr = np.uint8(color.rgb2ycbcr(imageRgb))
y2,cb,cr = cv2.split(imageYcbcr)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(cr,cmap=cm.Greys_r)
axarr[0,1].imshow(v,cmap=cm.Greys_r)
plt.show()
