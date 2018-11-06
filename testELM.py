from scipy import misc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from hpelm import ELM
import math
import cv2
import numpy as np


elm = ELM(1, 2)
X = np.array([-1, -0.6, -0.3, 0.3, 0.6, 1])
T = np.array([[1, 0],
              [1, 0],
              [1, 0],
              [0, 1],
              [0, 1],
              [0, 1]])
elm.add_neurons(100, "sigm")
elm.train(X, T, 'c')
print(elm.predict(X))