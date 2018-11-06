#!/usr/bin/env python

from scipy import misc
from decimal import *
from skimage import color
from skimage.measure import regionprops
from enum import Enum
from numba import jit
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys

""" Classe para segmentacao da folha """
class SegmentaFolha:

    """ Construtor """
    def __init__(self,imageRgb,canal):
        self.imageRgb = imageRgb
        self.plantaSegmentada = None
        self.plantaMascara = None
        self.canal = canal

    def getImageRgb(self):
        return self.imageRgb

    def getPlantaSegmentada(self):
        return self.plantaSegmentada

    def getPlantaMascara(self):
        return self.plantaMascara

    def getCanal(self):
        return self.canal

    """ Metodo da segmentacao """
    def segmenta(self):

        """ Aplica gaussian """
        imageRgb = cv2.GaussianBlur(self.imageRgb, (5, 5), 0)

        if (self.canal == 0):
            """ Fundo preto, azul e branco """
            imageHsv = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2HSV)
            imageYcbcr = np.uint8(color.rgb2ycbcr(self.imageRgb))
            """ Split """
            h, s, v = cv2.split(imageHsv)
            #plt.imshow(v,cmap=cm.Greys_r)
            #plt.show()
            y, cb, cr = cv2.split(imageYcbcr)
            #cv2.imwrite("/home/giuliano/Documentos/h.png",h)
            #cv2.imwrite("/home/giuliano/Documentos/s.png", s)
            #cv2.imwrite("/home/giuliano/Documentos/v.png", v)
            #cv2.imwrite("/home/giuliano/Documentos/cb.png", cb)
            #cv2.imwrite("/home/giuliano/Documentos/cr.png", cr)
            #print("fim do salvar")
            self.segmentaCbS(cb,s)
        elif (self.canal == 1):
            """ Fundo preto, azul """
            imageYcbcr = np.uint8(color.rgb2ycbcr(self.imageRgb))
            y, cb, cr = cv2.split(imageYcbcr)
            self.segmentaCb(cb)
        elif (self.canal == 2):
            """ Fundo preto, azul """
            imageYcbcr = np.uint8(color.rgb2ycbcr(self.imageRgb))
            y, cb, cr = cv2.split(imageYcbcr)
            self.segmentaCr(cr)
        elif (self.canal == 3):
            imageHsv = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(imageHsv)
            self.segmentaH(h)
        elif (self.canal == 4):
            imageHsv = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(imageHsv)
            self.segmentaS(s)
        else:
            imageHsv = cv2.cvtColor(imageRgb, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(imageHsv)
            self.segmentaV(v)
            """Falta funcao aqui"""

    def segmentaCbS(self,cb,s):

        print("Cbs")
        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th1 = cv2.threshold(cb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, th2 = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th2 = np.logical_not(th2)
        th2 = np.uint8(th2)
        th = np.bitwise_and(th1, th2)
        th = np.logical_not(th)
        th = np.uint8(th)


        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)

        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255

        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        th = np.logical_not(th)

        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        img = np.logical_not(img)

        print("Fim")
        plt.imshow(img, cmap=cm.Greys_r)
        plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img

    def segmentaH(self,h):
        print("H")
        #plt.imshow(h, cmap=cm.Greys_r)
        #plt.show()

        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th = cv2.threshold(h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = np.logical_not(th)
        th = np.uint8(th)

        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)

        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255

        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        th = np.logical_not(th)

        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        img = np.logical_not(img)

        print("Fim")
        #plt.imshow(img, cmap=cm.Greys_r)
        #plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img

    def segmentaS(self,s):
        print("S")
        #plt.imshow(s, cmap=cm.Greys_r)
        #plt.show()

        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = np.logical_not(th)
        th = np.uint8(th)

        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)

        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255

        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        th = np.logical_not(th)

        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        #img = np.logical_not(img)

        print("Fim")
        #plt.imshow(img, cmap=cm.Greys_r)
        #plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img

    def segmentaV(self,v):
        print("V")
        #plt.imshow(v, cmap=cm.Greys_r)
        #plt.show()

        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = np.logical_not(th)
        th = np.uint8(th)

        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)

        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255

        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        th = np.logical_not(th)

        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        #img = np.logical_not(img) COMENTADO PARA FUNDO PRETO

        print("Fim")
        #plt.imshow(img, cmap=cm.Greys_r)
        #plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img

    def segmentaCb(self,cb):
        print("Cb")

        #plt.imshow(cb, cmap=cm.Greys_r)
        #plt.show()

        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th = cv2.threshold(cb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = np.logical_not(th)
        th = np.uint8(th)

        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)

        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255

        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        th = np.logical_not(th)

        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        img = np.logical_not(img)

        print("Fim")
        #plt.imshow(img, cmap=cm.Greys_r)
        #plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img

    def segmentaCr(self, cr):
        print("Cr")

        #plt.imshow(cr,cmap=cm.Greys_r)
        #plt.show()

        """ Kernel 3,3 """
        kernel1 = (3, 3)
        kernel1 = np.ones(kernel1, np.uint8)

        ret, th = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #th = np.logical_not(th)
        th = np.uint8(th)


        # Operacao de dilatacao para corrigir possiveis falhas na segmentacao.
        # E preparar para remocao dos ruidos.
        th = cv2.erode(th, kernel1, iterations=3)


        # Remove os ruidos do fundo. Pixels brancos que nao estao ligados a planta.
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=4)
        sizes = stats[:, -1]

        max_label = 1
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        th = np.zeros(output.shape)
        th[output == max_label] = 255


        # Fecha alguns focos de preto no interior da folha. Que foram mal segmentados.
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1, iterations=10)
        #th = np.logical_not(th) -- COMENTADO PARA FUNDO AZUL


        # Remove focos de preto que continuaram no interior da folha.
        for i in xrange(th.shape[0]):
            th[i][0] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(th))
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[1]
        for i in xrange(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        img = np.zeros(output.shape)
        img[output == max_label] = 255
        img = np.logical_not(img)

        print("Fim")
        #plt.imshow(img, cmap=cm.Greys_r)
        #plt.show()

        # Plant with disease
        self.plantaSegmentada = cv2.bitwise_and(self.imageRgb, self.imageRgb, mask=np.uint8(img))
        self.plantaMascara = img











