from eval_utils import load_results as data
from eval_utils.export_eval import json_dump, results_table_dump

current_classifiers = [clf['name'] for clf in data.config_dict['CLASSIFIERS']]


def calc_and_dump(data_dict, func, filename, title):
    processed_results_dict = calc_stats(data_dict, func)
    json_dump(processed_results_dict, filename, 'results/')
    results_table_dump(processed_results_dict, filename, title)


def calc_stats(dict, func):

    stat_dict = {}

    # For each label, run the average
    for label, seed_dict in dict.items():

        label_stat_dict = get_label_stat(seed_dict, func)

        # We build the output dict label-by-label
        stat_dict.update({label: label_stat_dict})

        # break

    return stat_dict


def get_label_stat(seed_dict, func):

    # Init template dictionary of metrics
    scores_template_dict = {k: [] for k in data.config_dict['METRICS']}
    # Init dictionary
    label_avg_dict = {k: scores_template_dict.copy()
                      for k in current_classifiers}

    # Build dictionary by adding scores into list
    for clf_dict in seed_dict.values():

        for clf, score_dict in clf_dict.items():

            for score, value in score_dict.items():

                label_avg_dict[clf][score].append(value)

    # Average each list item
    for clf, score_dict in label_avg_dict.items():

        for score, values in score_dict.items():

            label_avg_dict[clf][score] = func(values)

    return label_avg_dict
