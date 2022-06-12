
import math
from multiprocessing.dummy import Array
from operator import index
from cv2 import sqrt
import numpy as np
import pandas as pd
import random
from collections import Counter

from scipy import rand

class Kmeans:
    def __init__(self, dataset, k):
        # Remove the last column (the target one)
        self.dataset = dataset.iloc[: , :-1]
        # Save the target row in a separate variable
        self.target = dataset.iloc[: , -1]
        self.k = k
        self.clusters = [0] * self.dataset.shape[0]
        
    def clusterize(self, showConfusion = False):
        centroids = self.initCentroids()
        quiescent = False
        while quiescent == False:
            print('QUI')
            oldClusters = self.clusters
            for index, row in zip(range(0,self.dataset.shape[0]), self.dataset.itertuples(index=False)):
                distances = []
                for center in centroids:
                    # If the centroid representing that group is set to None it means there was 
                    # no element part of that group
                    if(center is not None):
                        #print('YELLO')
                        distances.append(self.calculateDistance(center, row))
                    else:
                        distances.append(None)
                self.clusters[index] = distances.index(min(dist for dist in distances if dist is not None))
            oldCentroids = centroids.copy()
            centroids = self.recalculateCentroids()
            quiescent = np.array_equal(centroids, oldCentroids)
        if(showConfusion):
            # TODO Implement confusion matrix
            print(Counter(self.target))
            print(np.bincount(self.clusters))
            df = pd.DataFrame({'Labels': self.target, 'Clusters': self.clusters})
            ct = pd.crosstab(df['Labels'], df['Clusters'])
            print(ct)
        return self.clusters
        
        
    def initCentroids(self):
        centroids = []
        random.seed = 10
        for i in range(0, self.k):
            index = random.randint(0, self.dataset.count(axis=0)[0])
            centroids.append(self.dataset.iloc[index])
        return centroids
    
    def recalculateCentroids(self):
        centroids = [None] * self.k
        numElem = [0] * self.k
        for cl, row in zip(self.clusters, self.dataset.itertuples(index=False)):
            if(centroids[cl] == None): 
                centroids[cl] = row
            else: 
                for center, rowEl in zip(centroids[cl], row): center += rowEl
            numElem[cl] += 1
        for center in centroids[cl]: center = center / numElem[cl]
        return centroids
    
    def calculateDistance(self, a, b):
        assert(len(a) == len(b))
        lenght = len(a)
        sum = 0
        for key in range(0, lenght):
            sum += pow(a[key] - b[key], 2)
        return math.sqrt(sum)
            