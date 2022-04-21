import copy
from pprint import pprint

from eval_utils import load_results as data

current_classifiers = [clf['name'] for clf in data.config_dict['CLASSIFIERS']]


def calc_stats(results_dict, row_headers, table_ids, match_clf):

    stat_dict = {}

    # For each table id, build a table
    for label, seed_dict in results_dict.items():
        # for label, seed_dict in results_dict.items():

        label_stat_dict = get_label_stats(seed_dict, row_headers, match_clf)

        # We build the output dict label-by-label
        stat_dict.update({label: label_stat_dict})

    # pprint(stat_dict)

    return stat_dict


def get_label_stats(seed_dict, row_headers, match_clf=True):

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

                # If clf is not in row_ids, which are the actual classifiers
                if clf not in row_ids:
                    continue  # Skip baseline classifiers

                for metric, score in score_dict.items():

                    # Collect a list of all relevant scores
                    label_stat_dict[clf][metric].append(score)

    else:

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                for metric, score in score_dict.items():

                    # Append for each row
                    [label_stat_dict[v][metric].append(score) for v in row_ids]

    # Apply function to each list item
    for row, score_dict in label_stat_dict.items():

        for metric, values in score_dict.items():

            # Get current function
            func = row_headers[row]
            # Apply function
            label_stat_dict[row][metric] = func(values)

    return(label_stat_dict)


def get_classifier_stats():
    pass
