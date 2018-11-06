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


NUM_ATTR = 17
class main:

    if __name__ == '__main__':

        #imageRgb = misc.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/branco/15.jpg")
        path = sys.argv[1]
        nomeImagem = sys.argv[2]
        canal = sys.argv[3]
        nameToSave = nomeImagem.split(".")

        imageRgb = misc.imread(path+nomeImagem)
        segmenta = SegmentaFolha(imageRgb,int(canal))
        segmenta.segmenta()

        #print("Segmentando lesoes com YCgCr...")
        #folhaSemLesoes, lesoes = SegmentaLesoes.method_ycgcr2(
        #    segmenta.getPlantaSegmentada(),
        #    segmenta.getPlantaMascara())
        #print("Lesoes segmentadas")

        #imageYcbcr = np.uint8(color.rgb2ycbcr(segmenta.getPlantaSegmentada()))
        #imageHsv = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2HSV)

        #y, cb, cr = cv2.split(imageYcbcr)
        #h, s, v = cv2.split(imageHsv)

        #plt.imshow(cb, cmap=cm.Greys_r)
        #plt.show()
        #plt.imshow(cr, cmap=cm.Greys_r)
        #plt.show()
        #plt.imshow(h, cmap=cm.Greys_r)
        #plt.show()
        #plt.imshow(s, cmap=cm.Greys_r)
        #plt.show()
        #plt.imshow(v, cmap=cm.Greys_r)
        #plt.show()


        folhaSemLesoes,lesoes = SegmentaLesoes.method_ycgcr2(segmenta.getPlantaSegmentada(),segmenta.getPlantaMascara())
        #folhaSemLesoes, lesoes = SegmentaLesoes.method_kmeans(segmenta.getPlantaSegmentada(),segmenta.getPlantaMascara())
        print("Lesoes segmentadas")

        #plt.imshow(segmenta.getPlantaMascara(),cmap=cm.Greys_r)
        #plt.show()

        #plt.imshow(folhaSemLesoes)
        #plt.show()

        #mask = np.asarray(mask,dtype=np.uint8)
        #mask = np.logical_and(mask,segmenta.getPlantaMascara())
        #mask = np.asarray(mask)
        #mask = mask.astype(int)

        #print(mask)
        #plt.imshow(mask, cmap=cm.Greys_r)
        #plt.show()

        #plt.imshow(folhaSemLesoes)
        #plt.show()

        """ Calcula a area da folha """
        areaFolha = cv2.countNonZero(segmenta.getPlantaSegmentada()[:,:,1])
        areaLesoes = areaFolha - cv2.countNonZero(folhaSemLesoes[:, :, 1])
        print(np.float32(areaLesoes)/np.float32(areaFolha)*100)
        #print(areaFolha)


        misc.imsave(path+nameToSave[0]+"_1.png",segmenta.getPlantaSegmentada())
        misc.imsave(path+nameToSave[0]+"_2.png",folhaSemLesoes)


        if (len(lesoes)!=0):
            extrair = ExtracaoCaracteristicas(lesoes)
            extrair.extracaoAtributos()

            scaler = joblib.load('slr.pkl')
            nn = joblib.load('nn.pkl')

            atributos = extrair.atributos
            for i in xrange(NUM_ATTR):
                atributos[:, i] = np.divide((
                        atributos[:, i] - scaler.data_min_[i]),
                        scaler.data_max_[i] - scaler.data_min_[i])

            classificacao = nn.predict(extrair.atributos)

            severidade0 = 0 # Bixo mineiro
            severidade1 = 0 # Ferrugem.

            for i in xrange(len(classificacao)):
                if (classificacao[i]==0):
                    severidade0 += cv2.countNonZero(lesoes[i][:,:,1])
                else:
                    severidade1 += cv2.countNonZero(lesoes[i][:, :, 1])


            #print("Classificacao:" + str(nn.predict(extrair.atributos)))
            print([
                round(np.float32(severidade0)/np.float32(areaFolha)*100,2),
                round(np.float32(severidade1)/np.float32(areaFolha)*100,2),
                round(np.float32(areaLesoes)/np.float32(areaFolha)*100,2)])

            #for i in xrange(len(lesoes)):
            #    plt.imshow(lesoes[i])
             #   plt.show()

        #plt.imshow(plantWd3)
        #plt.show()




