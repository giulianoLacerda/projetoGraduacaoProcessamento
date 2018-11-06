from scipy import misc
from decimal import *
from skimage import color
from skimage.measure import regionprops
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import sys
import SegmentaLesoes
import correction
import ExtracaoAtributos as ea
from sklearn.externals import joblib


def teste():

    path = "/home/giuliano/Documentos/UFES_2018_1/P.G/branco_lesoes_classes/"
    numeroClasses = 2
    percentTreino = 0.7
    data = []
    totalTreino = []
    dataNormalizado = []
    taxaAprend = [0.001,0.01,0.1,0.2,0.3]
    neuronios = [10,20,30,40,50,100,200]
    decaimento = ["constant","invscaling"]


    #data = np.genfromtxt(path+'data.csv', delimiter=',')
    for i in xrange(numeroClasses):
        data.append(np.genfromtxt(path+'data'+str(i)+'.csv', delimiter=','))
        totalTreino.append(int(round(len(data[i])*percentTreino,0)))

    for z in xrange(len(decaimento)):

        for k in xrange(len(taxaAprend)):

            for w in xrange(len(neuronios)):
                print("------------------------------------")
                print("Para neuronios:"+str(neuronios[w]))
                print("Para taxa:"+str(taxaAprend[k]))
                print("Para decaimento:"+decaimento[z])
                recall = 0
                precision = 0
                acuracia = 0

                r = []
                p = []
                a = []

                recall2 = 0
                precision2 = 0
                acuracia2 = 0

                for j in xrange(0,100):

                    for i in xrange(numeroClasses):
                        np.random.shuffle(data[i])
                        #print(i)
                        #data[i][:,1:] = dataNormalizado[i]
                        #print(data[i])

                    dataTreino = np.concatenate((data[0][0:totalTreino[0],:],data[1][0:totalTreino[1],:]),axis=0)

                    dataTeste = np.concatenate((data[0][totalTreino[0]:,:],data[1][totalTreino[1]:,:]),axis=0)

                    # tuple (min, max), default=(0, 1)
                    scaler = MinMaxScaler((0,1))
                    scaler.fit(dataTreino[:,1:])

                    #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

                    # Normaliza entre 0 e 1
                    for i in xrange(17):
                        dataTreino[:,i+1] = np.divide((dataTreino[:,i+1]-scaler.data_min_[i]),scaler.data_max_[i]-scaler.data_min_[i])
                        dataTeste[:,i+1] = np.divide((dataTeste[:,i+1]-scaler.data_min_[i]),scaler.data_max_[i]-scaler.data_min_[i])

                    #np.savetxt("dataTreino.csv", dataTreino, delimiter=",")


                    #print(dataTreino)

                    # Embaralha
                    np.random.shuffle(dataTreino)
                    np.random.shuffle(dataTeste)

                    #dataTreino[:,1:] = scaler.transform(dataTreino[:,1:])
                    #dataTeste[:,1:] = scaler.transform(dataTeste[:,1:])

                    #print(dataTreino)
                    #print(dataTeste)

                    # Neural Network
                    nn = MLPClassifier(solver='sgd',
                                       hidden_layer_sizes=(neuronios[w]),
                                       activation='relu',
                                       learning_rate_init=taxaAprend[k],
                                       learning_rate=decaimento[z],
                                       max_iter=10000)

                    nn.fit(dataTreino[:,1:],dataTreino[:,0])

                    #print("Treino:"+str(nn.score(dataTreino[:,1:],dataTreino[:,0])))
                    #print("Validacao:"+str(nn.score(dataTeste[:,1:],dataTeste[:,0])))
                    predicao = nn.predict(dataTeste[:,1:])
                    #print(j)

                    precision = precision + precision_score(dataTeste[:,0],predicao)
                    recall = recall + recall_score(dataTeste[:,0],predicao)
                    acuracia = acuracia + nn.score(dataTreino[:,1:],dataTreino[:,0])

                    p.append(precision)
                    r.append(recall)
                    a.append(acuracia)

                    if j !=0:
                        #print(str(p[j])+"e"+str(p[j-1]))
                        #print(abs(p[j] - p[j-1]))
                        precision2 = precision2 + abs(p[j] - p[j-1])
                        recall2 = recall2 + abs(r[j] - r[j-1])
                        acuracia2 = acuracia2 + abs(a[j] - a[j-1])


                print("Acuracia:"+str(acuracia)+"+-"+str(acuracia2/100))
                print("Precision:"+str(precision)+"+-"+str(precision2/100))
                print("Recall:"+str(recall)+"+-"+str(recall2/100))
                print("------------------------------------")

                #joblib.dump(nn,'nn.pkl')
                #joblib.dump(scaler,'slr.pkl')
                #scaler2 = joblib.load('slr.pkl')
                #nn2 = joblib.load('nn.pkl')


teste()