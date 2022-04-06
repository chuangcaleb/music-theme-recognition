import matplotlib.pyplot as plt
from sklearn import tree
import json
import pickle

models_pickle = pickle.load(
    open("data/output/output_best_models.pickle", "rb"))
feature_names = json.load(
    open("data/output/final_feature_names.json", "r"))


def plotDecisionTree(estimator, feature_names, target_label):

    # Figure size
    plt.figure(figsize=(15, 9))
    # plt.figure(figsize=(20, 10))

    # Main plot
    tree.plot_tree(estimator, feature_names=feature_names, class_names=[
        'absent', target_label], label='root', filled=True, fontsize=5)

    # Labels
    plt.plot(target_label, label=target_label)
    plt.plot('absent', label='absent')
    plt.legend(loc='upper right')

    # Title
    # plt.title(
    #     f"\'{target_label}\' with DecisionTreeClassifier(max_leaf_nodes={best_max_leaf_nodes}, random_state={rand_state})")
    plt.title(f"\'{target_label}\'")

    plt.show()


# * FOR EACH LABEL -------------------------------------------------------------

for current_label in models_pickle:

    for clf in models_pickle[current_label]:

        if clf == 'DecisionTree':

            model = models_pickle[current_label][clf]
            plotDecisionTree(model, feature_names, current_label)

            # print(export_text(model, feature_names))
