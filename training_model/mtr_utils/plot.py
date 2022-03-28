from sklearn import tree
import matplotlib.pyplot as plt

# print(export_text(classifier, feature_names=final_columns))


def plotDecisionTree(estimator, feature_names, target_label, best_max_leaf_nodes, rand_state):

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
    plt.title(
        f"\'{target_label}\' with DecisionTreeClassifier(max_leaf_nodes={best_max_leaf_nodes}, random_state={rand_state})")

    plt.show()
