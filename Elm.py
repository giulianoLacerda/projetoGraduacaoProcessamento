from scipy import misc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from hpelm import ELM
import math
import cv2
import numpy as np


def teste():
    path = "/home/giuliano/Documentos/UFES_2018_1/P.G/branco_lesoes_classes/"
    numeroAtributos = 17
    numeroClasses = 2
    percentTreino = 0.7
    data = []
    totalTreino = []
    dataNormalizado = []


    #data = np.genfromtxt(path+'data.csv', delimiter=',')
    for i in xrange(numeroClasses):
        data.append(np.genfromtxt(path+'data'+str(i)+'.csv', delimiter=','))
        totalTreino.append(int(round(len(data[i])*percentTreino,0)))

    recall = 0
    precision = 0
    acuracia = 0
    for j in xrange(0,100):

        for i in xrange(numeroClasses):
            np.random.shuffle(data[i])


        dataTreino = np.concatenate((data[0][0:totalTreino[0],:],data[1][0:totalTreino[1],:]),axis=0)
        dataTeste = np.concatenate((data[0][totalTreino[0]:,:],data[1][totalTreino[1]:,:]),axis=0)

        scaler = MinMaxScaler((0,1))
        scaler.fit(dataTreino[:,1:])

        # Normaliza entre 0 e 1
        for i in xrange(numeroAtributos):
            dataTreino[:,i+1] = np.divide((dataTreino[:,i+1]-scaler.data_min_[i]),scaler.data_max_[i]-scaler.data_min_[i])
            dataTeste[:,i+1] = np.divide((dataTeste[:,i+1]-scaler.data_min_[i]),scaler.data_max_[i]-scaler.data_min_[i])

        # Embaralha
        np.random.shuffle(dataTreino)
        np.random.shuffle(dataTeste)
        y = []

        elm = ELM(numeroAtributos,1)
        elm.add_neurons(200, "tanh")
        #print(dataTreino[:,0])
        elm.train(dataTreino[:,1:], dataTreino[:,0])
        pr = elm.predict(dataTreino[:,1:])
        #print(dataTreino[:,0])
        #print(1-elm.error(dataTreino[:,0],pr))
        acuracia = acuracia + 1 - elm.error(dataTreino[:, 0], pr)

        pr = elm.predict(dataTeste[:,1:])

        for i in xrange(pr.shape[0]):
            if(pr[i]>0.5):
                y.append(1)
            else:
                y.append(0)

        precision = precision + precision_score(dataTeste[:,0],y)
        recall = recall + recall_score(dataTeste[:,0],y)

    print("Acuracia:"+str(acuracia/100))
    print("Precision:"+str(precision/100))
    print("Recall:"+str(recall/100))
    #print(y)