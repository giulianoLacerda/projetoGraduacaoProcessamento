from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

image = misc.imread('cafe2.jpg')

# Simple Average
# This formula is very simple:
# G = (R+G+B)/3

# Let's write a simple one-line method tha takes in a 
# RGB pixel and emits the average of that

def average(pixel):
    return (pixel[0]+pixel[1]+pixel[2])/3

# OR, even more, use numpy's average
# np.average(pixel)

# Weighted Average
# This method is more 'human' - because humans perceive some colors more
# than the rest, we apply a weighted average. Here's the formula:

# G = R*0.299 + G*0.587 + B*0.114

# Green is the most prominent color(nearly 60%), followed by red(30%) and
# finally blue(11%).

# Here's the equivalent code

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

# Conversion
# Now let's actually do the conversion

grey = np.zeros((image.shape[0], image.shape[1])) # init 2d numpy array

# get row number

for rownum in range(len(image)):
    for colnum in range(len(image[rownum])):
        grey[rownum][colnum] = weightedAverage(image[rownum][colnum])

# average or weightedAverage

# We have to specify to pyplot that it is a grayscale image,
# not a color image. We do that by using matplotlib.cm.Greys_, as
# shown below.

plt.imshow(grey,cmap=cm.Greys_r)
plt.show()

misc.imsave('cafe2_gray.jpg',grey)

plt.imshow(image) #load
plt.show() #show the window


#print image

print image.shape
