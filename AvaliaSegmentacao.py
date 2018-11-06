from SegmentaFolha import *
from ExtracaoCaracteristicas import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from scipy import misc
import SegmentaLesoes
import matplotlib.pyplot as plt
import sys
import cv2

NUM_ATTR = 41
class main:

    if __name__ == '__main__':

        a = [22,30,40,50,100,135,172,219,220,229]
        #b = [0,1,2,3,4,5]
        #c = ["CbS","Cb","Cr","H","S","V"]

        #for i in xrange(len(a)):

            #maskPS = cv2.cvtColor(maskPS, cv2.COLOR_BGR2GRAY)
            #maskPS = np.logical_not(maskPS)
            #maskPS = np.logical_not(maskPS)

            #plt.imshow(maskPS, cmap=9999cm.Greys_r)
            #plt.show()

        for j in xrange(len(a)):
            print("------------Imagem "+str(a[j])+"--------------")

            maskPS = cv2.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + ".png", 0)


            #maskPS = misc.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/"+ str(a[j])+".png")
            #maskPS = cv2.cvtColor(maskPS, cv2.COLOR_RGB2GRAY)
            maskSeg = cv2.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + "_3.png", 0)
            #maskSeg = misc.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + "_3.png")
            #maskSeg = cv2.cvtColor(maskSeg, cv2.COLOR_RGB2GRAY)

            #segmenta = SegmentaFolha(imagemRgb, b[j])
            #segmenta.segmenta()
            #misc.imsave("/home/giuliano/Documentos/UFES_2018_1/P.G/pretoMask/"+str(a[i])+"_"+c[j]+".png", np.asarray(segmenta.getPlantaMascara()).astype(int))

            #print(maskPS)
            #print(segmenta.getPlantaMascara())

            #if(maskPS.shape[0]!=segmenta.getPlantaMascara().shape[0]):
            #    maskPS = np.transpose(maskPS)

            result1 = np.logical_or(maskPS,maskSeg)
            result2 = np.logical_and(maskPS,maskSeg)
            result1 = np.asarray(result1)
            result2 = np.asarray(result2)
            result1 = result1.astype(int)
            result2 = result2.astype(int)

            den = cv2.countNonZero(result1)
            num = cv2.countNonZero(result2)

            x = cv2.countNonZero(maskSeg)
            y = cv2.countNonZero(maskPS)
            print(x)
            print(y)

            print(round(np.float32(num)/np.float32(den)*100,2))
            print(round((np.float32(y) / np.float32(x)) * 100, 2))

            print("----------------------------------------------")

            #plt.imshow(result.astype(int),cmap=cm.Greys_r)
            #plt.show()










