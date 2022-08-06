
from ast import And
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
import seaborn as sns

from scipy import rand

class Kmeans:
    def __init__(self, dataset, k):
        # Remove the last column (the target one)
        self.dataset = dataset.iloc[: , :-1]
        # Save the target row in a separate variable
        self.target = dataset.iloc[: , -1]
        self.k = k
        self.clusters = [0] * self.dataset.shape[0]
        self.centroids = []
        
    def clusterize(self):
        self.centroids = self.initCentroids()
        quiescent = False
        while not quiescent:
            for index, row in zip(range(0,self.dataset.shape[0]), self.dataset.itertuples(index=False)):
                distances = []
                for center in self.centroids:
                    # If the centroid representing that group is set to None it means there was 
                    # no element part of that group
                    if(center is not None):
                        distances.append(self.calculateDistance(center, row))
                    else:
                        distances.append(None)
                self.clusters[index] = distances.index(min(dist for dist in distances if dist is not None))
            oldCentroids = self.centroids.copy()
            self.centroids = self.recalculateCentroids()
            quiescent = np.array_equal(self.centroids, oldCentroids)
            
        cost = self.optimizFunc()

        return self.clusters, cost
        
    def initCentroids(self):
        centroids = []
        selected = []
        for i in range(0, self.k):
            index = random.choice([j for j in range(self.dataset.count(axis=0)[0]) if j not in selected])
            centroids.append(self.dataset.iloc[index])
            selected.append(index)
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
    
    def optimizFunc(self):
        n = self.dataset.count(axis=0)[0]
        acc = 0
        for index, row in zip(range(0,self.dataset.shape[0]), self.dataset.itertuples(index=False)):
            acc += self.calculateDistance(row, self.centroids[self.clusters[index]])**2
        return acc / n
    
    def calcPlotSilhouete(self):
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
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = []
        if self.dataset.shape[0] > 200:
            sample_silhouette_values = silhouette_samples(self.dataset, self.clusters)
        else:
            sample_silhouette_values = self.silhouetteCoeff()
        
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = self.meanSilhouette(sample_silhouette_values)
        print(
            "For k clusters =",
            self.k,
            "The average silhouette_score is :",
            silhouette_avg,
        )
        
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
        
    def silhouetteCoeff(self):
        silhouettes = np.array([]) 
        for i in range(self.dataset.shape[0]):
            a = self.silhCoeffA(i)
            b = self.silhCoeffB(i)
            if a < b:
                silhouettes = np.append(silhouettes, 1-(a/b))
            elif a == b:
                silhouettes = np.append(silhouettes, 0)
            else:
                silhouettes = np.append(silhouettes, (b/a)-1)
        return silhouettes
        
    def silhCoeffA(self, index):
        count = 0
        sum = 0
        for j in range(self.dataset.shape[0]):
            if index != j and self.clusters[index] == self.clusters[j]:
                sum += self.calculateDistance(self.dataset.iloc[index], self.dataset.iloc[j])
                count += 1
        return sum / (count - 1)

    def silhCoeffB(self, index):
        min = float("inf")
        for cl in range(self.k):
            if cl != self.clusters[index]:
                count = 0
                sum = 0
                tmp = 0
                for j in range(self.dataset.shape[0]):
                    if index != j and self.clusters[j] == cl:
                        sum += self.calculateDistance(self.dataset.iloc[index], self.dataset.iloc[j])
                        count += 1
                tmp = sum / count
                if tmp < min:
                    min = tmp
        return min

    def meanSilhouette(self, silhouettes):
        sum = 0
        for elem in silhouettes:
            sum += elem
        return sum / len(silhouettes)
            
    def calcExtValidation(self):
        print(Counter(self.target))
        print(np.bincount(self.clusters))
        df = pd.DataFrame({'Labels': self.target, 'Clusters': self.clusters})
        ct = pd.crosstab(df['Labels'], df['Clusters'])
        print(ct)
        self.assignClusters(ct)
        self.calcPrecRecall()
        
    def assignClusters(self, confusionMatrix):
        wrongClustered = 0
        sum = 0
        for i in range(confusionMatrix.shape[0]):
            clusterNum = list(confusionMatrix.iloc[i][confusionMatrix.iloc[i] == max(confusionMatrix.iloc[i])].items())[0][0]
            clusterLabel = confusionMatrix.head().index[i]
            print("Cluster",clusterNum,":",clusterLabel)
            for j in range(len(confusionMatrix.iloc[i])):
                if(j != clusterNum):
                    wrongClustered += confusionMatrix.iloc[i][j]
                else:
                    sum += confusionMatrix.iloc[i][j]
        print("\nIncorrectly clustered instances:", wrongClustered)
        print("Purity:", sum / self.dataset.shape[0])
    
    def calcPrecRecall(self):
        numCols = self.dataset.shape[1]
        tpSum = 0
        tnSum = 0
        fpSum = 0
        fnSum = 0
        for i in range(self.dataset.shape[0]):
            for j in range(i+1, self.dataset.shape[0]):
                if self.target[i] == self.target[j] and self.clusters[i] == self.clusters[j]:
                    tpSum = tpSum + 1
                elif self.target[i] != self.target[j] and self.clusters[i] != self.clusters[j]:
                    tnSum = tnSum + 1
                elif self.target[i] != self.target[j] and self.clusters[i] == self.clusters[j]:
                    fpSum = fpSum + 1
                else:
                    fnSum = fnSum + 1
        den = math.comb(self.dataset.shape[0], 2)
        TP = tpSum / den
        TN = tnSum / den
        FP = fpSum / den
        FN = fnSum / den
        rand_index = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1measure = (2*(precision*recall)) / (precision+recall)
        print("Accuracy:", rand_index)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F score:", f1measure)
        
    def plotScatter(self):
        sns.pairplot(self.dataset.assign(hue=self.target), hue="hue")