import matplotlib.pyplot as plt
from sklearn import tree

from eval_utils.load_data import load_json, load_pickle

root_path = "data/output/"
run_id = "without-threshold"

models_dict = load_pickle(root_path + run_id + "/output_best_models.pickle")
feature_names = load_json(root_path + run_id + "/final_feature_names.json")
results_dict = load_json(root_path + run_id + "/output_best_results.json")


def plotDecisionTree(estimator, feature_names, target_label):

    # Get scores object
    scores_dict = results_dict[target_label]['DecnTree']
    scores_list = [k + ' = ' + str(round(v, 3))
                   for k, v in scores_dict.items()]
    textstr = '\n'.join(scores_list)

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

for current_label in models_dict:

    for clf in models_dict[current_label]:

        if clf == 'DecnTree':

            model = models_dict[current_label][clf]
            plotDecisionTree(model, feature_names, current_label)

            # print(export_text(model, feature_names))
