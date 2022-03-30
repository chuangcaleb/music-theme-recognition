""" Configuration settings for the running the MTR model """

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


RAND_STATE = 2

TARGET_LABEL = 'risk'
# feature_start = 'Vertical_Unisons'
# feature_end = 'Minor_Major_Triad_Ratio'

K_VALUE = 3

SELECTED_LABELS = [
    # 'risk', 'contentment'
    'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom'
]

# * Feature Selection

THRESHOLD_VAL = 0

# * Cross-Validation Tuning

CV = 5
SCORE = ['f1_macro', 'precision', 'recall']

# * kNN

# KNN_PARAMETERS = {'k': range(3, 5), 's': [0.5, 0.7, 1.0]}
k_range = list(range(1, 31))
KNN_PARAMETERS = dict(n_neighbors=k_range)

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
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}

# * classifiers object

classifiers = [
    {
        'name': 'kNN',
        'model': KNeighborsClassifier(),
        'param': KNN_PARAMETERS
    },
    {
        'name': 'Decision Tree',
        'model': DecisionTreeClassifier(),
        'param': DT_PARAMETERS
    },
    {
        'name': 'SVM',
        'model': SVC(),
        'param': SV_PARAMETERS
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(),
        'param': RF_PARAMETERS
    },
]
