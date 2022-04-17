import json
import matplotlib.pyplot as plt
from eval_utils import config as cfg

label_stats_dict = json.load(open('data/labels/label_stats_summary.json', "r"))
std_results_dict = json.load(
    open(cfg.OUTPUT_PATH + 'results/best_results.json', "r"))

label_stats_list = [key for key, value in label_stats_dict['%'].items()]
label_stats_list.reverse()

color_list = ['r', 'b', 'g', 'c', 'm', 'y']

# print(std_results_dict['risk'])

# clf_results = dict.fromkeys(std_results_dict)

clf_list = ['GaussianNB', 'kNN', 'SVM', 'DecnTree', 'RandForest']
clf_results = {k: [] for k in clf_list}

for label, clf_dict in std_results_dict.items():

    for clf, scores_dict in clf_dict.items():
        if clf not in clf_list:
            continue

        clf_results[clf].append(scores_dict['accura'])

plt.title("Standard Deviation by Positive Label Proportion")
plt.xlabel("Positive Label Proportion (%)")
plt.ylabel("Standard Deviation")

for i, clf in enumerate(clf_list):
    plt.plot(label_stats_list,
             clf_results[clf], color=color_list[i], label=clf_list[i])

plt.xticks(rotation=90)
plt.tight_layout()
plt.legend()

plt.show()
