import json
import matplotlib.pyplot as plt
from eval_utils import config as cfg

label_stats_dict = json.load(open('data/labels/label_stats_summary.json', "r"))
std_results_dict = json.load(
    open(cfg.RUN_DIR + 'results/results_avg.json', "r"))
config_dict = json.load(
    open(cfg.RUN_DIR + 'run_config.json', "r"))

label_stats_list = [i for i in label_stats_dict['%'].keys()]
label_stats_list.reverse()
# label_stats_list = config_dict['SELECTED_LABELS']

label_y = 'ROC-AUC'

color_list = ['r', 'b', 'g', 'c', 'm', 'y']

clf_list = config_dict['ACTUAL_CLASSIFIERS']
clf_results = {k: [] for k in clf_list}

for label, clf_dict in std_results_dict.items():

    for clf, scores_dict in clf_dict.items():

        if clf not in clf_list:
            continue

        clf_results[clf].append(scores_dict['rocauc'])

plt.title(label_y + " by Positive Label Proportion")
plt.xlabel("Positive Label Proportion (%)")
plt.ylabel(label_y)

for i, clf in enumerate(clf_list):
    plt.plot(label_stats_list,
             clf_results[clf],
             color=color_list[i],
             label=clf_list[i],
             marker='x'
             )


plt.xticks(rotation=90)
plt.axhline(y=0.5, linestyle='dashed', color='r', label='worthless test')
plt.axhline(y=0.6, linestyle='dashed', color='r',
            alpha=0.4, label='ROC-AUC = 0.6')
plt.tight_layout()
plt.legend(fontsize='7')

plt.show()
