# First: sudo -H pip install colormath

from colormath.color_objects import LabColor, XYZColor, HSLColor, sRGBColor
from colormath.color_conversions import convert_color
from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

imageRgb = misc.imread('./Imgs/aires1.jpg')
imageLab = np.zeros((imageRgb.shape[0], imageRgb.shape[1])) # init 2d numpy array

plt.imshow(imageLab)
plt.show()

print (len(imageRgb))

for rownum in range(len(imageRgb)):
    for colnum in range(len(imageRgb[rownum])):
        #print (imageRgb[rownum][colnum])
        rgb = sRGBColor(imageRgb[rownum][colnum][0],imageRgb[rownum][colnum][1],imageRgb[rownum][colnum][2])
        lab = convert_color(rgb,LabColor)
        print(lab.get_value_tuple())



plt.imshow(imageLab)
plt.show()
