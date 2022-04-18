from tabulate import tabulate
import pandas as pd

from eval_utils import load_results as data
from eval_utils.export_eval import tables_txt_dump
import matplotlib.pyplot as plt

label_df = pd.read_excel('data/labels/song_theme_label_database.xlsx')
label_df = label_df[label_df.recognizable == 1]
label_df.reset_index(drop=True, inplace=True)


models_pickle = data.models_dict
feature_list = data.feature_list

all_tables = {}

print("\n\n========================== Feature Importance scores per label ==========================\n")

for current_label in models_pickle:

    label_title = f'\n> \033[93m{current_label}\033[0m'
    scores_dict = data.best_results_dict[current_label]['RandForest']
    scores_str = [k + ' = ' + str(round(v, 3))
                  for k, v in scores_dict.items()]
    textstr = '\n'.join(scores_str)

    # print(label_title, end='\n\n')
    # print(textstr, end='\n\n')

    for clf in models_pickle[current_label]:

        if clf == 'RandForest':
            model = models_pickle[current_label][clf]

    tree_feature_importances = model.feature_importances_

    # Get index of scores, reverse sorted according to scores
    sorted_idx = tree_feature_importances.argsort()[::-1]

    top_features_scores = []

    for id in sorted_idx[:10]:
        top_features_scores.append([feature_list[id], round(
            tree_feature_importances[id], 3)])

    table = tabulate(top_features_scores, headers=[
        'feature', 'imp_sc'], tablefmt='github')

    print(table, end='\n\n')

    all_tables.update({current_label: table})


print()

# tables_txt_dump(all_tables, 'Feature Importances', 'md/feat_imp.md')

for label, table in all_tables.items():

    pos_label_i = label_df.index[label_df[label] == 1].tolist()
    print(pos_label_i)
    # for feat in table:
    #     plt.plot()

    break
