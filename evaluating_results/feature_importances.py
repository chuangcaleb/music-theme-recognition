from matplotlib import pyplot as plt
import numpy as np
from eval_utils.load_data import load_json, load_pickle
from tabulate import tabulate

root_path = "data/output/"
run_id = "temp"

models_dict = load_pickle(root_path + run_id + "/output_best_models.pickle")
feature_names = load_json(root_path + run_id + "/final_feature_names.json")


def printFeatureImportances(models_pickle, feature_names):

    for current_label in models_pickle:

        print(f'\n> \033[93m{current_label}\033[0m')

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
                    # print(feature_names[id], end=": \t")
                    # print(round(tree_feature_importances[id], 3))
                    top_features_scores.append([feature_names[id], round(
                        tree_feature_importances[id], 3)])

                table = tabulate(top_features_scores, headers=[
                    'feature', 'imp_sc'], tablefmt='github')

                print(table + '\n')

    print()


printFeatureImportances(models_dict, feature_names)
