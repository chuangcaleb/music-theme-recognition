""" Configuration settings for the running the MTR model """

import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# * MODEL PARAMS ---------------------------------------------------------------

# * Random Seed

RAND_SEED = 899
NUM_OF_RAND_SEEDS = 3

# List of random seeds
random.seed(RAND_SEED)
RAND_SEEDS_LIST = random.sample(range(1, 999999), NUM_OF_RAND_SEEDS)
# print(RAND_STATE_RANGE)

BEST_SEED_SCORING = 'f1-sc'

# * Label Selection

TARGET_LABEL = 'risk'

SELECTED_LABELS = [
    'risk', 'contentment',
    # 'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom'
]

# * Feature Selection

THRESHOLD_VAL = 0

# * Cross-Validation Tuning

CV = 5

CV_SCORING = 'f1_macro'

# * CLASSIFIERS ----------------------------------------------------------------

# * kNN

KNN_PARAMETERS = {
    'n_neighbors': list(range(1, 10)),
    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    # 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
}

# * Decision Tree

DT_PARAMETERS = {
    'max_leaf_nodes': range(3, 15),
    'criterion': ["gini", "entropy"]
}

# * SVM

SV_PARAMETERS = {'C': [0.1, 1, 10, 100, 1000],
                 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                 'kernel': ['rbf']}

# * Random Forest

RF_PARAMETERS = {
    # 'n_estimators': [200, 300, 400],
    'n_estimators': [200, 350],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy']
}

# * Neural Network

NN_PARAMETERS = {
    # 'solver': ['lbfgs'],
    'max_iter': [1500, 1750, 2000],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': 10.0 ** -np.arange(1, 10),
    'hidden_layer_sizes': np.arange(10, 15),
}

# * Naive Bayes

NB_PARAMETERS = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# * classifiers object

classifiers = [
    {
        'name': 'kNN',
        'model': KNeighborsClassifier(),
        'param': KNN_PARAMETERS
    },
    {
        'name': 'DecisionTree',
        'model': DecisionTreeClassifier(),
        'param': DT_PARAMETERS
    },
    {
        'name': 'SVM',
        'model': SVC(),
        'param': SV_PARAMETERS
    },
    # {
    #     'name': 'RandomForest',
    #     'model': RandomForestClassifier(),
    #     'param': RF_PARAMETERS
    # },
    # {
    #     'name': 'NeuralNetwork',
    #     'model': MLPClassifier(),
    #     'param': NN_PARAMETERS
    # },
    # {
    #     'name': 'NaiveBayes',
    #     'model':  GaussianNB(),
    #     'param': NB_PARAMETERS
    # },

]
