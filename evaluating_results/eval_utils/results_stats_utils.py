import copy
from pprint import pprint
from statistics import mean, stdev
from tkinter import Label

from eval_utils import load_results as data

# from eval_utils.export_eval import json_dump, results_table_dump

current_classifiers = [clf['name'] for clf in data.config_dict['CLASSIFIERS']]


# def calc_and_dump(data_dict, func, filename, title):
#     processed_results_dict = calc_stats(data_dict, func)
#     dump_results(processed_results_dict)


# class Animal(enum.Enum):
#     dog = 1
#     cat = 2
#     lion = 3


def calc_stats(results_dict, row_headers, match_clf):

    stat_dict = {}

    # For each table id, build a table
    for label, seed_dict in results_dict.items():

        label_stat_dict = get_label_stat(seed_dict, row_headers, match_clf)

        # We build the output dict label-by-label
        stat_dict.update({label: label_stat_dict})

    # pprint(stat_dict)

    return stat_dict


def get_label_stat(seed_dict, row_headers, match_clf=True):

    row_ids = row_headers.keys()
    # row_funcs = list(row_headers.values())
    # print(row_funcs)

    # init template dict of metrics
    scores_template_dict = {k: [] for k in data.config_dict['METRICS']}
    # Init dictionary
    label_stat_dict = {k: copy.deepcopy(scores_template_dict)
                       for k in row_ids}

    # * Build dictionary of lists of scores
    if match_clf:

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                for score, value in score_dict.items():

                    label_stat_dict[clf][score].append(value)

    else:

        for clf_dict in seed_dict.values():

            for clf, score_dict in clf_dict.items():

                for score, value in score_dict.items():

                    [label_stat_dict[v][score].append(value) for v in row_ids]

    # Average each list item
    for row, score_dict in label_stat_dict.items():

        for score, values in score_dict.items():

            # print(row_funcs[row](values))
            # # print(label_stat_dict[row][score])

            label_stat_dict[row][score] = row_headers[row](values)

    return(label_stat_dict)
