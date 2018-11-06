from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load Image.

imageRgb = misc.imread('cafe7.jpg')
grey = np.zeros((imageRgb.shape[0], imageRgb.shape[1])) # init 2d numpy array

# Convert image from RGB to Lab color space.

imageLab = color.rgb2lab(imageRgb,'D65','10')
l,a,b = cv2.split(imageLab)

#lab_image = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2LAB)
#l,a,b = cv2.split(lab_image)

#print imageLab
#print l

# L=L*255/100; a=a + 128; b=b + 128
#for rownum in range(len(l)):
#    for colnum in range(len(l[rownum])):
#        l[rownum][colnum] = l[rownum][colnum]*255/100


#for rownum in range(len(a)):
#    for colnum in range(len(a[rownum])):
#        a[rownum][colnum] = a[rownum][colnum]+128

#for rownum in range(len(b)):
#    for colnum in range(len(b[rownum])):
#        b[rownum][colnum] = b[rownum][colnum]+128

plt.imshow(l,cmap=cm.Greys_r)
plt.show()

plt.imshow(a,cmap=cm.Greys_r)
plt.show()

plt.imshow(b,cmap=cm.Greys_r)
plt.show()

misc.imsave('cafe7_l.jpg',l)
misc.imsave('cafe7_a.jpg',a)
misc.imsave('cafe7_b.jpg',b)