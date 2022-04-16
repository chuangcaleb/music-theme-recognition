from tabulate import tabulate
from eval_utils import load_data as data


def printFeatureImportances(models_pickle, feature_list):

    print("\n\n========================== Feature Importance scores per label ==========================\n")

    for current_label in models_pickle:

        label_title = f'\n> \033[93m{current_label}\033[0m'
        scores_dict = data.results_dict[current_label]['DecnTree']
        scores_list = [k + ' = ' + str(round(v, 3))
                       for k, v in scores_dict.items()]
        textstr = '\n'.join(scores_list)

        print(label_title, end='\n\n')
        print(textstr, end='\n\n')

        for clf in models_pickle[current_label]:

            if clf == 'RandForest':
                model = models_pickle[current_label][clf]

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

        print(table, end='\n\n')

    print()


printFeatureImportances(data.models_dict, data.feature_list)
