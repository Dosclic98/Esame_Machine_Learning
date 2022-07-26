from ast import And
import math
from multiprocessing.dummy import Array
from operator import index
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

from scipy import rand

 
 
class BayesianLearner:
    # We suppose to have the target in the last column
    def __init__(self, dataset):
        # Remove the last column (the target one)
        self.dataset = dataset.iloc[: , :-1]
        # Save the target row in a separate variable
        self.target = dataset.iloc[: , -1]
        self.classNames = self.target.unique()
        # Initialize the matrixes containing means and variaces
        self.mus = [[0 for _ in range(self.dataset.shape[1])]] * len(self.classNames)
        self.vars = [[0 for _ in range(self.dataset.shape[1])]] * len(self.classNames)
        self.datasetByClass = []
        for className in self.classNames:
            # Get the index of the cases having a certain class
            indexes = self.target.index[self.target == className]
            classDf = self.dataset.iloc[indexes]
            self.datasetByClass.append(classDf)
        
    def learn(self):
        for i in range(len(self.classNames)):
            dataInClass = self.datasetByClass[i]
            for j in range(dataInClass.shape[1]):
                ithCol = dataInClass.iloc[:,j]
                self.mus[i][j] = np.mean(ithCol)
                self.vars[i][j] = np.var(ithCol)
    
    def predict(self, case, className):
        if className not in self.classNames:
            raise Exception("Unknown class name used")
        index = np.where(self.classNames == className)[0][0]
        
        dataInClass = self.datasetByClass[index]
        prior = dataInClass.shape[0] / self.dataset.shape[0]
        prod = 1
        for featureIndex in range(len(case)):
            prod *= ((1 / (math.sqrt(2*math.pi*self.vars[index][featureIndex])))*(math.exp(-( math.pow(case[featureIndex] - self.mus[index][featureIndex], 2) / (2*self.vars[index][featureIndex]) ))))
        prod *= prior
        
        return prod    
        