from ast import And
from imghdr import tests
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
import statistics

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
        self.mus = [[0 for _ in range(self.dataset.shape[1])] for _ in range(len(self.classNames))] 
        self.vars = [[0 for _ in range(self.dataset.shape[1])] for _ in range(len(self.classNames))]
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
                self.mus[i][j] = statistics.mean(ithCol)
                self.vars[i][j] = statistics.variance(ithCol)
    
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
    
    def evaluate(self, testSet):
        # Remove the last column (the target one)
        datasetTest = testSet.iloc[: , :-1]
        # Save the target row in a separate variable
        targetTest = testSet.iloc[: , -1]
        
        predictions = []
        for j in range(datasetTest.shape[0]):
            bestProb = -1
            bestI = 0
            for i in range(len(self.classNames)):
                prob = self.predict(datasetTest.iloc[j], self.classNames[i])
                if prob > bestProb:
                    bestProb = prob
                    bestI = i
            predictions.append(self.classNames[bestI])
        
        self.calcStatistics(targetTest, predictions)
        
    
    def calcStatistics(self, targetValues, predictions):
        print(Counter(targetValues))
        print(pd.value_counts(predictions))
        df = pd.DataFrame({'Labels': targetValues, 'Predicted': predictions})
        ct = pd.crosstab(df['Labels'], df['Predicted'])
        corrClass = 0
        wrongClass = 0
        for i in self.classNames:
            for j in self.classNames:
                if i != j:
                    wrongClass += ct[i][j]
                else:
                    corrClass += ct[i][j]
        print(ct)
        print("Correctly classified instances:",corrClass)
        print("incorrectly classified instances:",wrongClass)
        self.calcPrecRecall(ct)
        
    def calcPrecRecall(self, confusionMatrix):
        for cl in self.classNames:
            print("=== Detailed Accuracy for class", cl, "===")
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            sum = 0
            for i in self.classNames:
                for j in self.classNames:
                    if i == cl and i == j:
                        TP += confusionMatrix[i][j]
                    if i != cl and i == j:
                        TN += confusionMatrix[i][j]
                    if i == cl and i != j:
                        FP += confusionMatrix[i][j]
                    if i != cl and i != j:
                        FN += confusionMatrix[i][j]
                    sum += confusionMatrix[i][j]
                
            accuracy = (TP+TN)/(TP+TN+FP+FN)
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            f1measure = (2*(precision*recall)) / (precision+recall)
            tpRate = TP / (TP + FN)
            fpRate = FP / (FP + TN)
            pe = (((TP+FN)/sum)*((TP+FP)/sum)) + (((TN+FP)/sum)*((TN+FN)/sum))
            kCoeff = (accuracy - pe) / (1 - pe)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F measure:", f1measure)
            print("True Positive Rate:", tpRate)
            print("False Positive Rate:", fpRate)
            print("K-Coefficent", kCoeff)