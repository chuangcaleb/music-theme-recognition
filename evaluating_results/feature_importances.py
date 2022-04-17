from tabulate import tabulate

from eval_utils import config as cfg
from eval_utils import load_data as data
from eval_utils.export_results import tables_txt_dump


def printFeatureImportances(models_pickle, feature_list):

    print("\n\n========================== Feature Importance scores per label ==========================\n")

    all_tables = {}

    for current_label in models_pickle:

        label_title = f'\n> \033[93m{current_label}\033[0m'
        scores_dict = data.best_results_dict[current_label]['DecnTree']
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

        all_tables.update({current_label: table})

    print()

    tables_txt_dump(all_tables, '/feat_imp', '.md')


printFeatureImportances(data.models_dict, data.feature_list)
