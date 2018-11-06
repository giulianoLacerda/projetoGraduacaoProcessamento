from scipy import misc
from skimage import color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

imageRgb = misc.imread("./Imgs/giu20.jpg")
# build a lookup table mapping the pixel values [0, 255] to
# their adjusted gamma values
gamma = 1.5
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255
                  for i in np.arange(0, 256)]).astype("uint8")

# apply gamma correction using the lookup table
imageEnhanced = cv2.LUT(imageRgb, table)
misc.imsave("testeCorrecao.png",imageEnhanced)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(imageRgb)
axarr[0,1].imshow(imageEnhanced)
plt.show()
