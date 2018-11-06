#!/usr/bin/env python

from scipy import misc
from scipy import stats
from skimage import color
from skimage.measure import regionprops, label
from skimage.measure import find_contours
from skimage.feature.texture import greycomatrix
from skimage.feature.texture import greycoprops
from skimage.filters.rank import entropy
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Retorna are, perimetro e diametro
def extracaoForma(arrayLesoes):
    print("Extraindo forma...")

    # Array com todos atributos
    atributosForma = []

    # Array de atributos de forma
    area = []
    perimetro = []
    diametro = []
    redondeza = []
    compactness = []

    arrayLesoesGrey = []

    for i in xrange(len(arrayLesoes)):

        # Converte para escala de cinza e encontra a mascara da lesao.
        arrayLesoesGrey.append(cv2.cvtColor(arrayLesoes[i], cv2.COLOR_RGB2GRAY))
        ret, maskb = cv2.threshold(arrayLesoesGrey[i], 0, 255, cv2.THRESH_BINARY)

        labelMaskb = label(maskb)
        propriedades = regionprops(labelMaskb)
        #plt.imshow(labelMaskb,cmap=cm.Greys_r)
        #plt.show()

        maxArea = 0
        maxPerimetro = 0
        maxDiametro = 0
        maxRedondeza = 0;
        maxCompactness = 0;

        for prop in propriedades:

            # Pega o label com maior area. Ignora os outros ruidos
            if(prop.area>maxArea):
                maxArea = round(prop.area,4)
                # Area
                #area.append(round(prop.area,2))
                #print("Area")
                #print(prop.area)

                # Perimetro
                maxPerimetro = round(prop.perimeter,4)
                #print(prop.perimeter)

                # Diametro
                maxDiametro = round(prop.major_axis_length,4)

                # Redondeza
                # Eh definida como a relacao entre a area da lesao
                # e a area de um circulo perfeito que contem a folha.
                maxRedondeza = round((100*maxArea)/(math.pi*((maxDiametro/2)**2)),4)

                # Compactness
                # Eh definida como a relacao entre o perimetro
                # e a area da lesao.
                maxCompactness = round((maxPerimetro**2)/maxArea,4)

        area.append(maxArea)
        perimetro.append(maxPerimetro)
        diametro.append(maxDiametro)
        redondeza.append(maxRedondeza)
        compactness.append(maxCompactness)

    atributosForma.append(area)
    atributosForma.append(perimetro)
    atributosForma.append(diametro)
    atributosForma.append(redondeza)
    atributosForma.append(compactness)

    #print(atributosForma)
    return atributosForma

def extracaoTextura(arrayLesoes):
    print("Extraindo texturas...")

    arrayLesoesGrey = []

    # Array com todos atributos de textura
    atributosTextura = []

    # Array de atributos de textura
    contraste = []
    dissimilaridade = []
    homogeneidade = []
    asm = []
    energia = []
    correlacao = []

    for i in xrange(len(arrayLesoes)):
        arrayLesoesGrey.append(cv2.cvtColor(arrayLesoes[i], cv2.COLOR_RGB2GRAY))
        #plt.imshow(arrayLesoesGrey[i],cmap=cm.Greys_r)
        #plt.show()

        # Offset 1 e angulo 0. O parametro symmetric eh true pois ele verifica as ocorrencias de
        # 1,2 e 2,1, ou seja, o equivalente a usar angulos de 0 e 180 e somar a GLCM.
        g = greycomatrix(arrayLesoesGrey[i], [1], [0], levels=256, normed = True, symmetric=True)

        # Elimina a coluna e linha 0, para que a glcm final nao tenha influencia do fundo.
        g = g[1:, 1:, :, :]


        # Atributos de textura.
        contraste.append(round(greycoprops(g, 'contrast').flatten()[0],4))
        dissimilaridade.append(round(greycoprops(g, 'dissimilarity').flatten()[0],4))
        homogeneidade.append(round(greycoprops(g,'homogeneity').flatten()[0],4))
        asm.append(round(greycoprops(g,'ASM').flatten()[0],4))
        energia.append(round(greycoprops(g,'energy').flatten()[0],4))
        correlacao.append(round(greycoprops(g,'correlation').flatten()[0],4))

    atributosTextura.append(contraste)
    atributosTextura.append(dissimilaridade)
    atributosTextura.append(homogeneidade)
    atributosTextura.append(asm)
    atributosTextura.append(energia)
    atributosTextura.append(correlacao)

    #print(atributosTextura)
    return atributosTextura

def extracaoCor(arrayLesoes):
    print("Extraindo cor...")

    atributosCor = []

    nElementos = len(arrayLesoes)
    arrayLesoesGrey = []
    arrayLesoesHSV = []

    # Array de atributos de cor
    mediaRGB = [[0 for x in range(3)] for y in range(nElementos)]
    desvioPadraoRGB = [[0 for x in range(3)] for y in range(nElementos)]
    varianciaRGB = [[0 for x in range(3)] for y in range(nElementos)]
    curtoseRGB = [[0 for x in range(3)] for y in range(nElementos)]
    entropiaRGB = [[0 for x in range(3)] for y in range(nElementos)]
    #assimetriaRGB = [][3]

    mediaHSV = [[0 for x in range(3)] for y in range(nElementos)]
    desvioPadraoHSV = [[0 for x in range(3)] for y in range(nElementos)]
    varianciaHSV = [[0 for x in range(3)] for y in range(nElementos)]
    curtoseHSV = [[0 for x in range(3)] for y in range(nElementos)]
    entropiaHSV = [[0 for x in range(3)] for y in range(nElementos)]
    #assimetriaHSV = [][3]

    # Percentual de cor vermelha
    #percntVermelho = [[0 for x in range(3)] for y in range(nElementos)]

    for i in xrange(nElementos):

        # Cacula a mascara.
        arrayLesoesGrey.append(cv2.cvtColor(arrayLesoes[i], cv2.COLOR_RGB2GRAY))
        ret, maskb = cv2.threshold(arrayLesoesGrey[i], 0, 255, cv2.THRESH_BINARY)

        # Converte para HSV
        arrayLesoesHSV.append(cv2.cvtColor(arrayLesoes[i], cv2.COLOR_RGB2HSV))

        for j in xrange(3):
            # Calcula o histograma.

            #plt.imshow(maskb,cmap=cm.Greys_r)
            #plt.show()
            histograma = cv2.calcHist([arrayLesoes[i]], channels=[j], mask=maskb, histSize=[256], ranges=[0,256])
            histograma2 = cv2.calcHist([arrayLesoesHSV[i]], channels=[j], mask=maskb, histSize=[256], ranges=[0, 256])

            # Media, variancia e desvioPadrao RGB
            (mediaRGB[i][j], desvioPadraoRGB[i][j]) = cv2.meanStdDev(arrayLesoes[i][:, :, j], mask=np.uint8(maskb))
            mediaRGB[i][j] = round(np.asscalar(mediaRGB[i][j][0]),4)
            desvioPadraoRGB[i][j] = round(np.asscalar(desvioPadraoRGB[i][j][0]),4)
            varianciaRGB[i][j] = round((desvioPadraoRGB[i][j])**2,4)
            #varianciaRGB[i][j] = np.asscalar(varianciaRGB[i][j][0])
            curtoseRGB[i][j] = stats.kurtosis(histograma)
            curtoseRGB[i][j] = round(np.asscalar(curtoseRGB[i][j][0]),4)
            entropiaRGB[i][j] = stats.entropy(histograma)
            entropiaRGB[i][j] = round(np.asscalar(entropiaRGB[i][j][0]),4)

            # Media, variancia e desvioPadrao HSV
            (mediaHSV[i][j], desvioPadraoHSV[i][j]) = cv2.meanStdDev(arrayLesoesHSV[i][:, :, j], mask=np.uint8(maskb))
            mediaHSV[i][j] = round(np.asscalar(mediaHSV[i][j][0]),4)
            desvioPadraoHSV[i][j] = round(np.asscalar(desvioPadraoHSV[i][j][0]),4)
            varianciaHSV[i][j] = round((desvioPadraoHSV[i][j])**2,4)
            #varianciaHSV[i][j] = np.asscalar(varianciaHSV[i][j][0])
            curtoseHSV[i][j] = stats.kurtosis(histograma2)
            curtoseHSV[i][j] = round(np.asscalar(curtoseHSV[i][j][0]),4)
            entropiaHSV[i][j] = stats.entropy(histograma2)
            entropiaHSV[i][j] = round(np.asscalar(entropiaHSV[i][j][0]),4)



    # Para cada atributo, os valores de cada lesao

    atributosCor.append(mediaRGB)
    atributosCor.append(desvioPadraoRGB)
    atributosCor.append(varianciaRGB)
    atributosCor.append(curtoseRGB)
    atributosCor.append(entropiaRGB)

    atributosCor.append(mediaHSV)
    atributosCor.append(desvioPadraoHSV)
    atributosCor.append(varianciaHSV)
    atributosCor.append(curtoseHSV)
    atributosCor.append(entropiaHSV)

    #print(atributosCor[4])

    return atributosCor


