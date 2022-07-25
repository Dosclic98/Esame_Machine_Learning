# -*- coding: utf-8 -*-
"""
Created on Sun Apr 3 15:15:00 2022

@author: Davide Savarro
"""

from calendar import c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from kmeans import Kmeans

def elbowMethod(dataset, maxClusters, maxIterations):
    costs = []
    for nCen in range(1, maxClusters+1):
        print("Calculating elbow for kClusters:", nCen)
        clustering, cost = findBestClusering(dataset, nCen, maxIterations)
        costs.append(cost)
    plt.plot(np.arange(1, maxClusters+1), np.array(costs))
    plt.xlabel("K clusters")
    plt.ylabel("Cost function")
    plt.title("Elbow method")
    plt.show()
    

def findBestClusering(dataset, kClusters, maxIterations):
    bestClustering = []
    bestCost = float("inf")
    for i in range(maxIterations):
        cl = Kmeans(dataset, kClusters)
        clusters, currCost = cl.clusterize()
        if currCost < bestCost:
            bestCost = currCost
            bestClustering = cl
    return bestClustering, bestCost


dataset = pd.read_csv("data/Data_for_UCI_named.csv")
# The p1 attribute is a non predictive one 
dataset = dataset.drop('p1', axis=1)

# I create two datasets where there are two different target attribute
datasetStab = dataset.drop('stabf', axis=1)
datasetStabF = dataset.drop('stab', axis=1)

cl, cost = findBestClusering(datasetStabF, 2, 5)
cl.calcExtValidation()
cl.calcPlotSilhouete()

dataset = pd.read_csv("data/iris.csv")
cl, cost = findBestClusering(dataset, 3, 10)
cl.calcExtValidation()
cl.calcPlotSilhouete()


elbowMethod(datasetStabF, 5, 5)
elbowMethod(dataset, 6, 10)

