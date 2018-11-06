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

        erroK=0
        erroY=0
        for j in xrange(len(a)):
            print("------------Imagem "+str(a[j])+"--------------")

            image = misc.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + "A"+str(a[j]) + ".png")
            r,g,b = cv2.split(image)
            areaFolha = cv2.countNonZero(g)


            maskPS = cv2.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + ".png",0)
            areaMaskP = cv2.countNonZero(maskPS)



            maskKmeans = cv2.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + "_2.png",0)
            areaMaskK = cv2.countNonZero(maskKmeans)


            maskY = cv2.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/brancoLesoesMask/" + str(a[j]) + "_3.png",0)
            areaMaskY = cv2.countNonZero(maskY)


            erroK = erroK + abs(1 - np.float32(areaFolha - areaMaskK) / np.float32(areaFolha - areaMaskP))
            erroY = erroY + abs(1 - np.float32(areaFolha - areaMaskY)/np.float32(areaFolha - areaMaskP))
            print("Severidade Photoshop:"+str(np.float32(areaFolha-areaMaskP)/np.float32(areaFolha)))
            print("Area Esperada:"+str(areaFolha-areaMaskP))
            print("Area Mask Kmeans:"+str(areaMaskK))
            print("Erro Kmeans:" + str(1 - np.float32(areaFolha - areaMaskK) / np.float32(areaFolha - areaMaskP)))
            print("Area Mask YCgCr:" + str(areaMaskY))
            print("Erro YCgCr:" + str(1 - np.float32(areaFolha - areaMaskY)/np.float32(areaFolha - areaMaskP)))



            #print("Severidade Kmeans:" + str(round(np.float32(areaFolha-areaMaskK)/np.float32(areaFolha),3)*100))
            #print("Severidade YCgCr:" + str(round(np.float32(areaFolha-areaMaskY)/np.float32(areaFolha),3)*100))


            print("----------------------------------------------")

        print("Erro Medio YCgCr:"+str(round(np.float(erroY)/10,3)))
        print("Erro Medio Kmeans:" + str(round(np.float(erroK) / 10, 3)))

            #plt.imshow(result.astype(int),cmap=cm.Greys_r)
            #plt.show()










