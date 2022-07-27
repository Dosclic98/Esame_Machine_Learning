from calendar import c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bayesianLearner import BayesianLearner

dataset = pd.read_csv("data/Data_for_UCI_named.csv")
# The p1 attribute is a non predictive one (calculated based on the other p attributes)
dataset = dataset.drop('p1', axis=1)
# Drop the other target column
dataset = dataset.drop('stab', axis=1)

datasetIris = pd.read_csv("data/iris.csv")
learner = BayesianLearner(datasetIris)
learner.learn()
learner.evaluate(datasetIris)