#!/usr/bin/python

from scipy import misc
from skimage import color
from skimage.measure import regionprops
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import SegmentaLesoes

def gammaCorrection(imageRgb):
	imageGrey = cv2.cvtColor(imageRgb,cv2.COLOR_RGB2GRAY)
	# Calcula o valor da intensidade media.
	media = np.mean(imageGrey)
	#print(media)
	
	# Normaliza o valor entre [0,1]
	maximo = np.amax(imageGrey)
	#print(maximo)
	valor = media/maximo	
	#print(valor)
	gamma = 1.0

	# Obtem o valor de gamma usando: 10*valor-4 -> valor>0.5 | 1 -> valor=0.5 | r^0.1-0.4 -> valor<0.5
	if valor>0.5:
		gamma = (10*valor)-4
	elif valor == 0.5:
		gamma = 1
	else:
		gamma = (valor**0.1)-0.1
	
	#print(gamma)
	# Aplica LU
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		              for i in np.arange(0, 256)]).astype("uint8")

	# apply gamma correction using the lookup table
	imageCorrigida = cv2.LUT(imageRgb, table)
	
	return imageCorrigida
