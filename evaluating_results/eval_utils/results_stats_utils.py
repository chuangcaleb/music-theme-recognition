from statistics import mean

# * Average --------------------------------------------------------------------


def average_results(dict):

    avg_results_dict = {}

    # For each label, run the average
    for label, seed_dict in dict.items():

        label_avg_dict = get_label_average(seed_dict)

        # We build the output dict label-by-label
        avg_results_dict.update({label: label_avg_dict})

        # break

    return avg_results_dict


def get_label_average(seed_dict):

    # Init dictionary
    label_avg_dict = init_label_average_dict(seed_dict)

    # Build dictionary by adding scores into list
    for seed, clf_dict in seed_dict.items():

        for clf, score_dict in clf_dict.items():

            for score, value in score_dict.items():

                label_avg_dict[clf][score].append(value)

    # Average each list item
    for clf, score_dict in label_avg_dict.items():

        for score, value in score_dict.items():

            label_avg_dict[clf][score] = mean(label_avg_dict[clf][score])

    return label_avg_dict


def init_label_average_dict(seed_dict):

    return_dictionary = {}

    for seed, clf_dict in seed_dict.items():

        for clf, score_dict in clf_dict.items():

            return_dictionary[clf] = {}

            for score in score_dict:

                return_dictionary[clf][score] = []

        break  # Only run this for the first seed

    return return_dictionary
