""" Configuration settings for the running the MTR model """

RAND_STATE = 2

TARGET_LABEL = 'risk'
# feature_start = 'Vertical_Unisons'
# feature_end = 'Minor_Major_Triad_Ratio'

K_VALUE = 3

SELECTED_LABELS = [
    'risk'
    # 'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom'
]

# * Feature Selection

threshold_val = 0

# * Cross-Validation Tuning

CV = 5
SCORE = 'f1_macro'

# * Decision Tree

DT_PARAMETERS = {
    'max_leaf_nodes': range(3, 15),
    'criterion': ["gini", "entropy"]
}
MAX_LEAF_NODES = 10

# * kNN

KNN_PARAMETERS = {'k': range(3, 5), 's': [0.5, 0.7, 1.0]}
