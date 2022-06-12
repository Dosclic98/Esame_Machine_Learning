# -*- coding: utf-8 -*-
"""
Created on Sun Apr 3 15:15:00 2022

@author: Davide Savarro
"""


import numpy as np
import pandas as pd

from kmeans import Kmeans

dataset = pd.read_csv("data/Data_for_UCI_named.csv")
# The p1 attribute is a non predictive one 
dataset = dataset.drop('p1', axis=1)

# I create two datasets where there are two different target attribute
datasetStab = dataset.drop('stabf', axis=1)
datasetStabF = dataset.drop('stab', axis=1)

#cl1 = Kmeans(datasetStab, 2)
#cl1.clusterize(showConfusion=False)

cl2 = Kmeans(datasetStabF, 2)
cl2.clusterize(showConfusion=True)