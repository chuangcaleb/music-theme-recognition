""" Configuration settings for the running the MTR model """

import random
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# * MODEL PARAMS ---------------------------------------------------------------

# * Random Seed

""" Random seed for the random generator """
RAND_SEED = 899
""" Number of random seeds to generate """
NUM_OF_RAND_SEEDS = 10

# List of random seeds
random.seed(RAND_SEED)
RAND_SEEDS_LIST = random.sample(range(1, 999999), NUM_OF_RAND_SEEDS)
# print(RAND_STATE_RANGE)

""" 
Scoring metric for selecting the best seed

Refer to: building_model/mtr_utils/scoring.py
"""
BEST_SEED_SCORING = 'f1-bin'

# * Label Selection

""" 
Specify labels to process or skip

Full list: 'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom' 
"""
SELECTED_LABELS = [
    # 'risk', 'contentment',
    'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom'
]

# * Feature Selection

""" 
Remove features with a variance below this value
"""
THRESHOLD_VAL = 0.005

# * Cross-Validation Tuning

""" 
Number of folds to use during cross-validation
"""
CV = 5

""" 
Scoring metric for selecting the best fold in cross-validation

Refer to: https://scikit-learn.org/stable/modules/model_evaluation.html
"""
BEST_CV_SCORING = 'f1'

# * CLASSIFIERS ----------------------------------------------------------------

# * kNN

KNN_PARAMETERS = {
    'n_neighbors': list(range(1, 10)),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
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
    'n_estimators': [100, 200, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy']
}

# * Neural Network

NN_PARAMETERS = {
    # 'solver': ['lbfgs'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [1500, 1750, 2000],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': 10.0 ** -np.arange(1, 10),
    'hidden_layer_sizes': np.arange(10, 15),
}

# * Naive Bayes

NB_PARAMETERS = {
    'var_smoothing': np.logspace(0, -9, num=100)
}

# * classifiers object


class defClf:

    zeroR = {
        'name': 'ZeroR',
        'model': DummyClassifier(strategy='most_frequent'),
        'param': {}
    }

    randomR = {
        'name': 'RandomR',
        'model': DummyClassifier(strategy='stratified'),
        'param': {}
    }

    knn = {
        'name': 'kNN',
        'model': KNeighborsClassifier(),
        'param': KNN_PARAMETERS
    }

    decnTree = {
        'name': 'DecnTree',
        'model': DecisionTreeClassifier(),
        'param': DT_PARAMETERS
    }

    svm = {
        'name': 'SVM',
        'model': SVC(),
        'param': SV_PARAMETERS
    }

    randForest = {
        'name': 'RandForest',
        'model': RandomForestClassifier(),
        'param': RF_PARAMETERS
    }

    neuralNet = {
        'name': 'NeuralNet',
        'model': MLPClassifier(),
        'param': NN_PARAMETERS
    }

    naiveBayes = {
        'name': 'GaussianNB',
        'model':  GaussianNB(),
        'param': NB_PARAMETERS
    }


# Comment out individual classifiers that you want to skip
classifiers = [
    defClf.zeroR,
    defClf.randomR,
    defClf.knn,
    defClf.decnTree,
    defClf.svm,
    defClf.randForest,
    # defaultClassifier.neuralNet,
    defClf.naiveBayes
]
