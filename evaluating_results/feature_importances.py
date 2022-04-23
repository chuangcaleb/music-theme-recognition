import textwrap
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import pandas as pd
from matplotlib import markers

from eval_utils import load_results as data
from eval_utils.export_eval import tables_txt_dump
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

from eval_utils import config as cfg

# Import and extract label_df
label_df = pd.read_excel('data/labels/song_theme_label_database.xlsx')
label_df = label_df[label_df.recognizable == 1]
label_df.reset_index(drop=True, inplace=True)

# Import and scale feature_df
feat_df = pd.read_csv('data/features/song_theme_feature_database.csv')
feat_df = feat_df.iloc[:, 1:]
# scaler = MinMaxScaler()
df_scaled = minmax_scale(feat_df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=feat_df.columns)

# Concat
joined_df = pd.concat([label_df, df_scaled], axis=1)

models_dict = data.models_dict
feature_list = data.feature_list

all_tables = {}
all_scores = {}

# print("\n\n========================== Feature Importance scores per label ==========================\n")

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
    table = tabulate(top_features_scores, headers=[
                     'feature', 'imp_sc'], tablefmt='github')
    print(table, end='\n\n')
    all_tables.update({current_label: table})

tables_txt_dump(all_tables, 'Feature Importances', 'md/feat_imp.md')

print()


for label, features in all_scores.items():

    fig, ax = plt.subplots(figsize=(7, 4))

    groups = joined_df.groupby(label)

    marker_types = [markers.CARETUPBASE, markers.CARETDOWNBASE]

    for i, (label_value, group_df) in enumerate(groups):

        y_group = []
        x_group = []

        for (feature, score) in features:

            feature_display = feature
            feature_display = textwrap.fill(feature_display, 24)

            y_feat = group_df[feature].tolist()
            x_feat = [feature_display] * len(y_feat)
            y_group.extend(y_feat)
            x_group.extend(x_feat)

        ax.scatter(y_group, x_group, marker=marker_types[i],
                   alpha=0.4, label=label_value)

    plt.title('Feature Distribution for ' + label)
    plt.tick_params(axis='y', which='major', labelsize=6)

    ax.set_ylabel('Feature', fontsize=10)
    ax.set_xlabel('Normalized Value', fontsize=10)

    plt.tight_layout()
    plt.legend()

    plt.show()

    # break
