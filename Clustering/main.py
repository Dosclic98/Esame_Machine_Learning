# -*- coding: utf-8 -*-
"""
Created on Sun Apr 3 15:15:00 2022

@author: Davide Savarro
"""


import numpy as np
import pandas as pd

dataset = pd.read_csv("data/Data_for_UCI_named.csv")
# The p1 attribute is a non predictive one 
dataset = dataset.drop('p1', axis=1)

# I create two datasets where there are two different target attribute
datasetStab = dataset.drop('stabf', axis=1)
datasetStabF = dataset.drop('stab', axis=1)

datasetStab.size

print(datasetStab.columns)