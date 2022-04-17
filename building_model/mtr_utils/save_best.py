from mtr_utils import config as cfg


def save_best_models(label_results_dict, label_models_dict):

    label_best_results_dict = {}
    label_best_models_dict = {}
    label_best_params_dict = {}

    # Only iterate through actual classifiers, skipping baseline classifeirs
    for clf in cfg.ACTUAL_CLASSIFIERS:

        # Init best score to 0
        best_score = 0

        for current_seed in label_results_dict:

            # Get current score in the BEST_SEED_SCORING metric
            current_score = label_results_dict[current_seed][clf][cfg.BEST_SEED_SCORING]

            # If currently a better score
            if current_score >= best_score:
                # Update new best score
                best_score = current_score

                # print(f'{clf}\'s new best seed is {current_seed}')

                # Save best results
                label_best_results_dict[clf] = label_results_dict[current_seed][clf]
                label_best_models_dict[clf] = label_models_dict[current_seed][clf]
                label_best_params_dict[clf] = label_models_dict[current_seed][clf].get_params(
                )

    return label_best_results_dict, label_best_models_dict, label_best_params_dict
