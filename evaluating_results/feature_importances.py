from ntpath import join
from tabulate import tabulate
import pandas as pd

from eval_utils import load_results as data
from eval_utils.export_eval import tables_txt_dump
import matplotlib.pyplot as plt

from eval_utils import config as cfg

label_df = pd.read_excel('data/labels/song_theme_label_database.xlsx')
label_df = label_df[label_df.recognizable == 1]
label_df.reset_index(drop=True, inplace=True)

prc_feat_df = pd.read_csv(cfg.RUN_DIR + '/processed_features.csv')

joined_df = label_df.join(prc_feat_df)

models_dict = data.models_dict
feature_list = data.feature_list

all_tables = {}
all_scores = {}

print("\n\n========================== Feature Importance scores per label ==========================\n")

for current_label in models_dict:

    # Grab variables
    label_title = f'\n> \033[93m{current_label}\033[0m'
    scores_dict = data.best_results_dict[current_label]['RandForest']
    scores_str = [k + ' = ' + str(round(v, 3))
                  for k, v in scores_dict.items()]
    textstr = '\n'.join(scores_str)

    # print(label_title, end='\n\n')
    # print(textstr, end='\n\n')

    # Grab
    for clf in models_dict[current_label]:

        if clf == 'RandForest':
            model = models_dict[current_label][clf]

    tree_feature_importances = model.feature_importances_

    # Get index of scores, reverse sorted according to scores
    sorted_idx = tree_feature_importances.argsort()[::-1]

    # Build feature lists
    top_features_scores = []

    # Report top 10 important features
    for id in sorted_idx[:10]:
        top_features_scores.append(
            (
                feature_list[id],
                round(tree_feature_importances[id], 3)
            )
        )
    all_scores.update({current_label: top_features_scores})

    # Report into tables
#     table = tabulate(top_features_scores, headers=[
#                      'feature', 'imp_sc'], tablefmt='github')
#     print(table, end='\n\n')
#     all_tables.update({current_label: table})

# tables_txt_dump(all_tables, 'Feature Importances', 'md/feat_imp.md')

print()


for label, features in all_scores.items():

    # pos_label_i = label_df.index[label_df[label] == 1].tolist()
    # neg_label_i = label_df.index[label_df[label] == 0].tolist()
    # neg_label_i = [x for x
    #                in list(range(len(label_df)))
    #                if not x in pos_label_i]

    fig, ax = plt.subplots()

    pos_feat_tuples = []
    neg_feat_tuples = []

    # for feat_pair in features:

    #     feature_name = feat_pair[0]

    # pos_feat_series = prc_feat_df.iloc[pos_label_i][feature_name]
    # pos_feat_list = pos_feat_series.tolist()
    # neg_feat_series = prc_feat_df.iloc[neg_label_i][feature_name]
    # neg_feat_list = neg_feat_series.tolist()

    # pos_feat_tuples.extend([(feature_name, y) for y in pos_feat_list])
    # neg_feat_tuples.extend([(feature_name, y) for y in neg_feat_list])

    # x_pos = [i[1] for i in pos_feat_tuples]
    # y_pos = [i[0] for i in pos_feat_tuples]
    # x_neg = [i[1] for i in pos_feat_tuples]
    # y_neg = [i[0] for i in pos_feat_tuples]
    groups = joined_df.groupby(label)

    for name, group in groups:

        y_group = []
        x_group = []

        for (feature, score) in features:

            y_feat = group[feature].tolist()
            x_feat = [feature] * len(y_feat)
            y_group.extend(y_feat)
            x_group.extend(x_feat)

        ax.scatter(y_group, x_group, marker='x')

    plt.title('Feature Distribution for ' + label)

    plt.tight_layout()

    plt.show()
