from SegmentaFolha import *
from ExtracaoCaracteristicas import *
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from scipy import misc
from tempfile import TemporaryFile
import SegmentaLesoes
import matplotlib.pyplot as plt
import sys

class main:

    if __name__ == '__main__':

        lesoes = []
        for i in xrange(1,257):
            imageRgb = misc.imread("/home/giuliano/Documentos/UFES_2018_1/P.G/branco_lesoes_classes/1/"+str(i)+".png")
            lesoes.append(imageRgb)

        extrair = ExtracaoCaracteristicas(lesoes)
        extrair.extracaoAtributos()
        atributos = extrair.atributos
        np.savetxt("data1.csv", atributos, delimiter=",")





