
import math
from multiprocessing.dummy import Array
from operator import index
from cv2 import sqrt
import numpy as np
import pandas as pd
import random
from collections import Counter
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from scipy import rand

class Kmeans:
    def __init__(self, dataset, k):
        # Remove the last column (the target one)
        self.dataset = dataset.iloc[: , :-1]
        # Save the target row in a separate variable
        self.target = dataset.iloc[: , -1]
        self.k = k
        self.clusters = [0] * self.dataset.shape[0]
        
    def clusterize(self, showConfusion = False, showSilhouette = False):
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
                        distances.append(self.calculateDistance(center, row))
                    else:
                        distances.append(None)
                self.clusters[index] = distances.index(min(dist for dist in distances if dist is not None))
            oldCentroids = centroids.copy()
            centroids = self.recalculateCentroids()
            quiescent = np.array_equal(centroids, oldCentroids)
        if(showConfusion):
            print(Counter(self.target))
            print(np.bincount(self.clusters))
            df = pd.DataFrame({'Labels': self.target, 'Clusters': self.clusters})
            ct = pd.crosstab(df['Labels'], df['Clusters'])
            print(ct)
        if(showSilhouette):
            # Create a subplot with 1 row and 2 columns
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(18, 7)

            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (k+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.dataset) + (self.k + 1) * 10])
            
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(self.dataset, self.clusters)
            print(
                "For k clusters =",
                self.k,
                "The average silhouette_score is :",
                silhouette_avg,
            )
            
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.dataset, self.clusters)
            
            y_lower = 10
            for i in range(self.k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                idx = []
                for j in range(len(self.clusters)):
                    if self.clusters[j] == i:
                        idx.append(j)
                ith_cluster_silhouette_values = sample_silhouette_values[idx]
                ith_cluster_silhouette_values.sort()
            

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / self.k)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.show()

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
            