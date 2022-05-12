""" Configuration settings for the building the MTR models """

import random

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mtr_utils.scaling import SCALERS

# * PATH -----------------------------------------------------------------------

RUN_ID = '.temp'

OUTPUT_PATH = 'data/output/' + RUN_ID + '/'


# * MODEL PARAMS ---------------------------------------------------------------


# * Random Seed

""" Random seed for the random generator """
RAND_SEED = 77

""" Number of random seeds to generate """
NUM_OF_RAND_SEEDS = 10


# List of random seeds
random.seed(RAND_SEED)
RAND_SEEDS_LIST = sorted(random.sample(range(1, 999999), NUM_OF_RAND_SEEDS))
# RAND_SEEDS_LIST = [50632] * 10

"""
Scoring metric for selecting the best seed

Refer to: building_model/mtr_utils/scoring.py
"""
BEST_SEED_SCORING = 'f1-bin'


# * Label Selection

"""
Specify labels to process or skip

Full list: 'grief', 'delusion', 'powerlessness', 'freedom', 'risk', 'safety', 'jadedness', 'authority', 'unity', 'celebration', 'contentment', 'love','desire', 'hope', 'wonder'
"""
SELECTED_LABELS = [
    # 'risk', 'contentment',
    'grief', 'delusion', 'powerlessness', 'freedom', 'risk', 'safety', 'jadedness', 'authority', 'unity', 'celebration', 'contentment', 'love', 'desire', 'hope', 'wonder'
]


# * Feature Engineering

"""
Remove features with a variance below this value
"""
THRESHOLD_VAL = 0


""" 
Options for scaling data

nrml = MinMaxScaler()
stnd = StandardScaler()
rbst = RobustScaler()
none = DummyScaler()

Refer to building_model/mtr_utils/scaling.py
"""
SCALER = SCALERS["stnd"]


# * Train-Test Split

"""
The proportion of dataset to include in the test set
"""
TEST_SIZE = 0.2


# * Cross-Validation Tuning

"""
Number of folds to use during cross-validation
"""
CV = StratifiedKFold(5)

"""
Scoring metric for selecting the best fold in cross-validation

Refer to: https://scikit-learn.org/stable/modules/model_evaluation.html
"""
BEST_CV_SCORING = 'f1'


# * Scoring

METRICS = [
    'f1-bin',
    'f1-mac',
    'accura',
    'precis',
    'recall',
    'rocauc'
]


# * CLASSIFIERS ----------------------------------------------------------------


# * Logistic Regression

_LR_PARAMETERS = {
    'C': np.logspace(-4, 4, 50),
    # 'penalty': ['l1', 'l2']
}

# * Naive Bayes

_NB_PARAMETERS = {
    'var_smoothing': np.logspace(0, -9, num=100)
}


# * kNN

_KNN_PARAMETERS = {
    'n_neighbors': list(range(1, 10)),
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
}


# * SVM

_SV_PARAMETERS = {'C': [0.1, 1, 10, 100],
                  'gamma': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                  'kernel': ['linear', 'poly', 'rbf']}


# * Decision Tree

_DT_PARAMETERS = {
    'max_leaf_nodes': range(3, 10),
    'criterion': ["gini", "entropy"],
    'splitter': ["best", "random"],
    'min_samples_split': range(2, 11, 3),
}

# * Random Forest

_RF_PARAMETERS = {
    # 'n_estimators': [200, 300, 400],
    'n_estimators': [75, 150, 220, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8],
    'criterion': ['gini', 'entropy']
}


# * Neural Network

_NN_PARAMETERS = {
    # 'solver': ['lbfgs', 'sgd', 'adam'],
    'solver': ['sgd', 'adam'],
    'max_iter': [200, 400],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': 10.0 ** -np.arange(1, 10),
    'hidden_layer_sizes': np.arange(10, 15),
}


# * classifiers object

""" Dict object to store all our classifiers neatly """
defaultClassifiers = {

    # Since we oversample and balance the training data, both classes will be equal. However, DummyClassifier still picks class '0' over class '1', even when they are equal. This is good, because '0' is our majority class.
    'zeroRate': {
        'name': 'ZeroRate',
        'code': 'zr',
        # 'model': DummyClassifier(strategy='constant', constant=0),
        'model': DummyClassifier(strategy='most_frequent'),
        'param': {}
    },
    'randomRate': {
        'name': 'RandomRate',
        'code': 'rr',
        'model': DummyClassifier(strategy='stratified'),
        'param': {}
    },
    'logisRegrs': {
        'name': 'LogisRegrs',
        'code': 'lr',
        'model': LogisticRegression(),
        'param': _LR_PARAMETERS
    },
    'naiveBayes': {
        'name': 'GaussianNB',
        'code': 'nb',
        'model':  GaussianNB(),
        'param': _NB_PARAMETERS
    },
    'knn': {
        'name': 'kNN',
        'code': 'kn',
        'model': KNeighborsClassifier(),
        'param': _KNN_PARAMETERS
    },
    'decnTree': {
        'name': 'DecnTree',
        'code': 'dt',
        'model': DecisionTreeClassifier(),
        'param': _DT_PARAMETERS
    },
    'svm': {
        'name': 'SVM',
        'code': 'sv',
        'model': SVC(),
        'param': _SV_PARAMETERS
    },
    'randForest': {
        'name': 'RandForest',
        'code': 'rf',
        'model': RandomForestClassifier(),
        'param': _RF_PARAMETERS
    },
    'neuralNet': {
        'name': 'NeuralNet',
        'code': 'nn',
        'model': MLPClassifier(),
        'param': _NN_PARAMETERS
    }
}


# Comment out individual classifiers that you want to skip
CLASSIFIERS = [
    # defaultClassifiers['zeroRate'],
    # defaultClassifiers['randomRate'],
    # defaultClassifiers['logisRegrs'],
    # defaultClassifiers['naiveBayes'],
    # defaultClassifiers['knn'],
    # defaultClassifiers['svm'],
    defaultClassifiers['decnTree'],
    # defaultClassifiers['randForest'],
    # defaultClassifiers['neuralNet'],
]

# ALL_CLASSIFIERS = [
#     clf['name'] for clf in CLASSIFIERS
# ]


ACTUAL_CLASSIFIERS = [
    clf['name'] for clf in CLASSIFIERS
    if type(clf['model']) is not DummyClassifier
]

# PIPELINE_MODEL_LIST = {
#     clf['code']: clf['model']
#     for clf in CLASSIFIERS
# }

# PIPELINE_PARAM_GRID = {
#     f"{clf['code']}__{k}": v
#     for clf in CLASSIFIERS
#     for k, v in clf['param'].items()
# }
