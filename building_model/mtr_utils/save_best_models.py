from mtr_utils import config as cfg


def save_best_models(label_results_dict, label_models_dict):

    label_best_results_dict = {}
    label_best_models_dict = {}

    for clf in cfg.classifiers:

        clf_name = clf['name']
        best_score = 0

        for current_seed in label_results_dict:

            current_score = label_results_dict[current_seed][clf_name][cfg.BEST_SEED_SCORING]

            if current_score >= best_score:

                label_best_results_dict[clf_name] = label_results_dict[current_seed][clf_name]
                label_best_models_dict[clf_name] = label_models_dict[current_seed][clf_name]

                best_score = current_score

                # print(f'{clf_name}\'s new best seed is {current_seed}')

    return label_best_results_dict, label_best_models_dict
