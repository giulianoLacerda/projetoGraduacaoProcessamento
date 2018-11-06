import cv2
import numpy as np
#optional argument
def nothing(x):
    pass
cv2.namedWindow('image')

#easy assigments
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'

cv2.createTrackbar(hl, 'image',0,179,nothing)
cv2.createTrackbar(hh, 'image',0,179,nothing)
cv2.createTrackbar(sl, 'image',0,255,nothing)
cv2.createTrackbar(sh, 'image',0,255,nothing)
cv2.createTrackbar(vl, 'image',0,255,nothing)
cv2.createTrackbar(vh, 'image',0,255,nothing)

frame=cv2.imread('aires1.jpg')
frame=cv2.GaussianBlur(frame,(5,5),0)
imS = cv2.resize(frame, (500, 300))

while(1):
    #convert to HSV from BGR
    hsv=cv2.cvtColor(imS, cv2.COLOR_BGR2YCR_CB)


    #read trackbar positions for all
    hul=cv2.getTrackbarPos(hl, 'image')
    huh=cv2.getTrackbarPos(hh, 'image')
    sal=cv2.getTrackbarPos(sl, 'image')
    sah=cv2.getTrackbarPos(sh, 'image')
    val=cv2.getTrackbarPos(vl, 'image')
    vah=cv2.getTrackbarPos(vh, 'image')
    #make array for final values
    HSVLOW=np.array([hul,sal,val])
    HSVHIGH=np.array([huh,sah,vah])

    #apply the range on a mask
    mask = cv2.inRange(hsv,HSVLOW, HSVHIGH)
    res = cv2.bitwise_and(imS,imS, mask =mask)

    cv2.imshow('image', res)
    cv2.imshow('yay', imS)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
