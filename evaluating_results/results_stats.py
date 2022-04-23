from statistics import mean, stdev

from eval_utils import load_results as data
from eval_utils.export_eval import dump_results
from eval_utils.results_stats_utils import (calc_clf_stats, calc_stats,
                                            get_label_stats, stats_range, stats_iqr)

current_classifiers = [clf['name'] for clf in data.config_dict['CLASSIFIERS']]
current_actual_classifiers = [
    clf for clf in data.config_dict['ACTUAL_CLASSIFIERS']]
current_labels = [clf for clf in data.config_dict['SELECTED_LABELS']]

average_clf = {clf: mean for clf in current_classifiers}
stdev_clf = {clf: stdev for clf in current_actual_classifiers}

stats = {
    'avg': mean,
    'best': max,
    'std': stdev,
    'rang': stats_range,
    'iqr': stats_iqr
}

# ---------------------------------------------------------------------------- #

avg_results_dict = calc_stats(
    data.all_results_dict, average_clf, match_clf=True)
dump_results(avg_results_dict, 'results_avg', 'Average')

std_results_dict = calc_stats(data.all_results_dict, stdev_clf, match_clf=True)
dump_results(std_results_dict, 'results_std', 'Standard Deviation')

label_results_dict = calc_stats(data.all_results_dict, stats, match_clf=False)
dump_results(label_results_dict, 'results_labels', 'Label Overview')

model_results_dict = calc_clf_stats(data.all_results_dict, stats)
dump_results(model_results_dict, 'results_models', 'Model Overview')
