from matplotlib import pyplot as plt
import numpy as np
from eval_utils.load_data import load_json, load_pickle
from tabulate import tabulate

root_path = "data/output/"
run_id = "with-threshold"

models_dict = load_pickle(root_path + run_id + "/output_best_models.pickle")
feature_list = load_json(root_path + run_id + "/final_feature_list.json")
# results_dict = load_json(root_path + run_id + "/output_best_results.json")


def printFeatureImportances(models_pickle, feature_list):

    print("\n\n========================== Feature Importance scores per label ==========================\n")

    for current_label in models_pickle:

        label_title = f'\n> \033[93m{current_label}\033[0m'

        print(label_title)

        for clf in models_pickle[current_label]:

            if clf == 'RandForest':

                print()

                model = models_pickle[current_label][clf]

                #
                tree_feature_importances = model.feature_importances_

                # Get index of scores, reverse sorted according to scores
                sorted_idx = tree_feature_importances.argsort()[::-1]

                top_features_scores = []

                for id in sorted_idx[:10]:
                    # print(feature_list[id], end=": \t")
                    # print(round(tree_feature_importances[id], 3))
                    top_features_scores.append([feature_list[id], round(
                        tree_feature_importances[id], 3)])

                table = tabulate(top_features_scores, headers=[
                    'feature', 'imp_sc'], tablefmt='github')

                print(table + '\n')

    print()


printFeatureImportances(models_dict, feature_list)
