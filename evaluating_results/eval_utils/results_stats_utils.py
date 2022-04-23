import copy

from eval_utils import load_results as data
import numpy as np

#! So messy. Needs rewriting!
actual_clf_list = data.config_dict['ACTUAL_CLASSIFIERS']


def calc_stats(results_dict, row_headers, match_clf):

    stat_dict = {}

    # For each label, build a table
    for label, seed_dict in results_dict.items():
        # for label, seed_dict in results_dict.items():

        label_stat_dict = get_label_stats(seed_dict, row_headers, match_clf)

        # We build the output dict label-by-label
        stat_dict.update({label: label_stat_dict})

    return stat_dict


def get_label_stats(seed_dict, row_headers, match_clf=True):

    # * Init output dictionary

    row_ids = row_headers.keys()

    # init template dict of metrics
    scores_template_dict = {k: [] for k in data.config_dict['METRICS']}
    # Init dictionary
    label_stat_dict = {k: copy.deepcopy(scores_template_dict)
                       for k in row_ids}

    # * Build dictionary of lists of scores
    if match_clf:

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                # If clf is not in row_ids, which vary
                if clf not in row_ids:
                    continue  # Skip baseline classifiers

                for metric, score in score_dict.items():

                    # Collect a list of all relevant scores
                    label_stat_dict[clf][metric].append(score)

    else:

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                # If clf is not in clf_list, which are the actual classifiers
                if clf not in actual_clf_list:
                    continue  # Skip baseline classifiers

                for metric, score in score_dict.items():

                    # Append for each row
                    [label_stat_dict[v][metric].append(score) for v in row_ids]

    # * Apply function to each list item

    for row, score_dict in label_stat_dict.items():

        for metric, values in score_dict.items():

            # Get current function
            func = row_headers[row]
            # Apply function
            label_stat_dict[row][metric] = func(values)

    return(label_stat_dict)


current_classifiers = [clf for clf in data.config_dict['ACTUAL_CLASSIFIERS']]


def calc_clf_stats(results_dict, row_headers):

    # * First we init the output dict

    row_ids = row_headers.keys()

    stat_dict = {}

    # init template dict of metrics
    scores_template_dict = {k: [] for k in data.config_dict['METRICS']}
    # Init dictionary
    clf_stat_dict = {k: copy.deepcopy(
        scores_template_dict) for k in row_ids}
    stat_dict = {clf: copy.deepcopy(
        clf_stat_dict) for clf in current_classifiers}

    # * Then we collect the input data

    # For each table id, build a table
    for label, seed_dict in results_dict.items():
        # for label, seed_dict in results_dict.items():

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                # If clf is not in row_ids, which are the actual classifiers
                if clf not in current_classifiers:
                    continue  # Skip baseline classifiers

                for metric, score in score_dict.items():

                    # Append for each row
                    [stat_dict[clf][v][metric].append(score) for v in row_ids]

    # * Now we apply the function to each item

    # For each table
    for clf_table, table_dict in stat_dict.items():

        # For each clf row
        for row, score_dict in table_dict.items():

            # For each metric column
            for metric, values in score_dict.items():

                # Get current function
                func = row_headers[row]
                # Apply function
                stat_dict[clf_table][row][metric] = func(values)

    return stat_dict


""" Calculates the range of values in a list """


def stats_range(data):
    range = max(data) - min(data)
    return range


""" Calculates the interquartile range of values in a list """


def stats_iqr(data):
    q3, q1 = np.percentile(data, [75, 25])
    iqr = q3 - q1
    return iqr
