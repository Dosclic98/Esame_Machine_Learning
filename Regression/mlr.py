import math
from multiprocessing.dummy import Array
from operator import index
import re
from statistics import variance
from cv2 import mean, sqrt
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
 
class MLR:
    def __init__(self, dataset, targetVarName):
         self.target = dataset.loc[:,targetVarName]
         self.dataset = dataset.drop(targetVarName, axis=1)
         # Add intercept
         self.dataset.insert(0, 'x0', 1)
    
    def learnParams(self, learningRate):
        self.params = pd.DataFrame(random.uniform(0,1), index=range(self.dataset.shape[1]), columns=range(1))
        mse = 0
        oldMse = 0
        deltaMse = 0
        isFirstRun = True
        m = self.dataset.shape[0]
        while deltaMse > 0.001 or isFirstRun:
            isFirstRun = False
            pred = self.predict(self.dataset, self.params)
            deltaParams = (learningRate/m) * np.dot((pred - self.target).transpose(), self.dataset.to_numpy()).transpose()
            self.params = self.params - deltaParams
            oldMse = mse
            mse = ((pred - self.target)**2).sum() / (2*m)
            dmse = oldMse - mse if not isFirstRun else mse
        return pred, self.params
        
    
    def predict(self, x, params):
        return x.to_numpy().dot(params)