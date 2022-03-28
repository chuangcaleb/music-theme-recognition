""" Configuration settings for the running the MTR model """

rand_state = 15

target_label = 'risk'
# feature_start = 'Vertical_Unisons'
# feature_end = 'Minor_Major_Triad_Ratio'

k_value = 3

selected_labels = [
    'love', 'contentment', 'desire', 'celebration', 'grief', 'unity', 'safety', 'risk', 'wonder', 'hope', 'jadedness', 'delusion', 'authority', 'powerlessness', 'freedom'
]

# * Feature Selection

threshold_val = 0

# * Cross-Validation Tuning

cv = 5
score = 'f1_macro'

# * Decision Tree

dt_parameters = {
    'max_leaf_nodes': range(3, 15),
    'criterion': ["gini", "entropy"]
}
max_leaf_nodes = 10

# * kNN

knn_parameters = {'k': range(3, 5), 's': [0.5, 0.7, 1.0]}
