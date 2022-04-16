from statistics import mean

# * Average --------------------------------------------------------------------


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

    # Init dictionary
    label_avg_dict = init_label_dict(seed_dict)

    # Build dictionary by adding scores into list
    for seed, clf_dict in seed_dict.items():

        for clf, score_dict in clf_dict.items():

            for score, value in score_dict.items():

                label_avg_dict[clf][score].append(value)

    # Average each list item
    for clf, score_dict in label_avg_dict.items():

        for score, value in score_dict.items():

            label_avg_dict[clf][score] = func(label_avg_dict[clf][score])

    return label_avg_dict


def init_label_dict(seed_dict):

    return_dictionary = {}

    for seed, clf_dict in seed_dict.items():

        for clf, score_dict in clf_dict.items():

            return_dictionary[clf] = {}

            for score in score_dict:

                return_dictionary[clf][score] = []

        break  # Only run this for the first seed

    return return_dictionary
