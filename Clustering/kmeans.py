
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
                    distances.append(self.calculateDistance(center, row))
                self.clusters[index] = distances.index(min(distances))
            self.recalculateCentroids()
            quiescent = np.array_equal(self.clusters, oldClusters)
        
        
    def initCentroids(self):
        centroids = []
        for i in range(0, self.k):
            index = random.randint(0, self.dataset.count(axis=0)[0])
            centroids.append(self.dataset.iloc[index])
        return centroids
    
    def recalculateCentroids(self):
        # TODO From here
    
    def calculateDistance(self, a, b):
        lenght = len(a)
        sum = 0
        for key in range(0, lenght):
            sum += pow(a[key] - b[key], 2)
        return math.sqrt(sum)
            