
import random

class Kmeans:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        
    def clusterize(self):
        centroids = self.initCentroids()
        
        
    def initCentroids(self):
        centroids = []
        for i in 0..self.k:
            index = random.randint(0, self.dataset.count(axis=0)[0])
            centroids.append(self.dataset.iloc[index])
        return centroids
    
    def calculateDistance(self, a, b):
        lenght = self.dataset.count(axis=0)[0]
        #for key in 0..lenght:
            