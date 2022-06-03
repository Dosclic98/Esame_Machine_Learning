
import math
from multiprocessing.dummy import Array
from cv2 import sqrt
import numpy as np
import random

class Kmeans:
    def __init__(self, dataset, k):
        # Remove the last column (the target one)
        self.dataset = dataset.iloc[: , :-1]
        # Save the target row in a separate variable
        self.target = dataset.iloc[: , -1]
        self.k = k
        self.clusters = [0] * self.dataset.shape[0]
        
    def clusterize(self):
        centroids = self.initCentroids()
        quiescent = False
        while quiescent == False:
            oldClusters = self.clusters
            for index, row in zip(range(0,self.dataset.shape[0]), self.dataset.itertuples()):
                distances = []
                for center in centroids:
                    # If the centroid representing that group is set to None it means there was 
                    # no element part of that group
                    if(center is not None):
                        distances.append(self.calculateDistance(center, row))
                    else:
                        distances.append(None)
                self.clusters[index] = distances.index(min(dist for dist in distances if dist is not None))
            oldCentroids = centroids.copy()
            self.recalculateCentroids()
            quiescent = np.array_equal(self.clusters, oldClusters)
        
        
    def initCentroids(self):
        centroids = []
        for i in range(0, self.k):
            index = random.randint(0, self.dataset.count(axis=0)[0])
            centroids.append(self.dataset.iloc[index])
        return centroids
    
    def recalculateCentroids(self):
        centroids = [None] * self.k
        numElem = [0] * self.k
        for cl, row in zip(self.clusters, self.dataset.itertuples()):
            if(centroids[cl] == None): centroids[cl] = row
            else: centroids[cl] += row
            numElem[cl] += 1
        # TODO Fix his division
        centroids[cl] = centroids[cl] / numElem[cl]
        return centroids
    
    def calculateDistance(self, a, b):
        lenght = len(a)
        sum = 0
        for key in range(0, lenght):
            sum += pow(a[key] - b[key], 2)
        return math.sqrt(sum)
            