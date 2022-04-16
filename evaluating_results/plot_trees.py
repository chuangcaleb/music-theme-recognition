import matplotlib.pyplot as plt
from sklearn import tree

from eval_utils import load_data as data

root_path = "data/output/"
run_id = "without-threshold"


def plotDecisionTree(estimator, feature_list, target_label):

    # Get scores object
    scores_dict = data.results_dict[target_label]['DecnTree']
    scores_list = [k + ' = ' + str(round(v, 3))
                   for k, v in scores_dict.items()]
    textstr = '\n'.join(scores_list)

    # Figure size
    plt.figure(figsize=(15, 9))
    # plt.figure(figsize=(20, 10))

    # Main plot
    tree.plot_tree(estimator, feature_names=feature_list, class_names=[
        'absent', target_label], label='root', filled=True, fontsize=5)

    # Labels
    plt.plot(target_label, label=target_label)
    plt.plot('absent', label='absent')
    plt.legend(loc='upper right')

    # Parameters in text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax = plt.gca()
    ax.text(0, 1, textstr,
            horizontalalignment='left',
            verticalalignment='top', fontsize=10, bbox=props, transform=ax.transAxes)

    # Title
    # plt.title(
    #     f"\'{target_label}\' with DecisionTreeClassifier(max_leaf_nodes={best_max_leaf_nodes}, random_state={rand_state})")
    plt.title(f"\'{target_label}\'")

    plt.show()


# * FOR EACH LABEL -------------------------------------------------------------

for current_label in data.models_dict:

    for clf in data.models_dict[current_label]:

        if clf == 'DecnTree':

            model = data.models_dict[current_label][clf]
            plotDecisionTree(model, data.feature_list, current_label)

            # print(export_text(model, feature_list))

    # break
