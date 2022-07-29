import math
from multiprocessing.dummy import Array
from operator import index
import re
from statistics import variance
from cv2 import mean, norm, sqrt
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
 
class MLR:
    def __init__(self, dataset, targetVarName, normMethod = "AUTO"):
        self.target = dataset[[targetVarName]]
        self.dataset = dataset.drop(targetVarName, axis=1)
        # Normalize with one of the following methods
        if normMethod == "AUTO":
            self.dataset = self.dataset.apply(lambda x: (x - x.mean()) / (x.std()))
        elif normMethod == "MINMAX":
            self.dataset = self.dataset.apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
        else:
            raise Exception("Unable to dine normalization method:", normMethod)
        # Adding Bias
        self.dataset.insert(0, 'x0', np.ones(self.dataset.shape[0]), True)
    
    def learnParams(self, learningRate):
        self.params = np.random.uniform(0,1,size=(self.dataset.shape[1], 1))#pd.DataFrame(random.uniform(0,1), index=range(self.dataset.shape[1]), columns=range(1))
        pred = np.empty((0,1), float)
        mse = 0
        oldMse = 0
        deltaMse = 0
        isFirstRun = True
        m = self.dataset.shape[0]
        while deltaMse > 0.001 or isFirstRun:
            pred = self.predict(self.dataset, self.params)
            deltaParams = (learningRate/m) * np.dot((pred - self.target).transpose(), self.dataset.values).transpose()
            self.params = self.params - deltaParams
            oldMse = mse
            mse = (((pred - self.target)**2).sum() / (2*m))[0]
            deltaMse = oldMse - mse if not isFirstRun else mse
            isFirstRun = False

        return pred, self.params
    
    # TODO Implement method to predict on a subset of data
        
    
    def predict(self, x, params):
        return np.dot(x.values, params)