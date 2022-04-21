from statistics import mean, stdev

from eval_utils import load_results as data
from eval_utils.results_stats_utils import calc_stats
from eval_utils.results_stats_utils import get_label_stats
from eval_utils.export_eval import dump_results

current_classifiers = [clf for clf in data.config_dict['ACTUAL_CLASSIFIERS']]
current_labels = [clf for clf in data.config_dict['SELECTED_LABELS']]

average_clf = {clf: mean for clf in current_classifiers}
stdev_clf = {clf: stdev for clf in current_classifiers}

stats = {
    'avrg': mean,
    'best': max,
    'stdv': stdev
}


# ---------------------------------------------------------------------------- #

avg_results_dict = calc_stats(
    data.all_results_dict, average_clf, match_clf=True)
dump_results(avg_results_dict, 'results_avg', 'Average')

std_results_dict = calc_stats(data.all_results_dict, stdev_clf, match_clf=True)
dump_results(std_results_dict, 'results_std', 'Standard Deviation')

label_results_dict = calc_stats(data.all_results_dict, stats, match_clf=False)
dump_results(label_results_dict, 'results_labels', 'Label Overview')

model_results_dict = calc_stats(data.all_results_dict, stats, match_clf=False)
dump_results(label_results_dict, 'results_models', 'Model Overview')
