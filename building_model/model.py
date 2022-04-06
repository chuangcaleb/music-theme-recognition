from sklearn.model_selection import train_test_split

from mtr_utils import config as cfg
from mtr_utils.export_results import (latextab_per_label, models_dump,
                                      results_dump, tables_dump)
from mtr_utils.feature_selection.auto_feature_selection import \
    filterVarianceThreshold
from mtr_utils.feature_selection.load_feature_set import \
    preselected_feature_list
from mtr_utils.import_dataset import raw_feature_df, raw_label_df
from mtr_utils.label_dataset_selection import extractLabelDataset
from mtr_utils.model_tuning import tuneClassifer
from mtr_utils.sampling import oversample, smote, undersample
from mtr_utils.save_best_models import save_best_models
from mtr_utils.scoring import get_scoring, round_scores

output_best_models_dict = {}
output_all_results_dict = {}
output_best_results_dict = {}

output_latex_tables = {}
output_md_tables = {}

# * Extract data from label dataset

label_df = extractLabelDataset(raw_label_df, cfg.SELECTED_LABELS)

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_list]

selected_feature_np, feature_names = filterVarianceThreshold(
    manual_feature_df, cfg.THRESHOLD_VAL)

# ? FEATURE ENGINEERING

# ?

# * FOR EACH LABEL -------------------------------------------------------------

for current_label in cfg.SELECTED_LABELS:

    print(f'\n\n> Building model for \033[92m{current_label}\033[0m...')

    label_results_dict = {}
    label_models_dict = {}

    # * FOR EACH RAND_SEED -----------------------------------------------------

    for current_seed in cfg.RAND_SEEDS_LIST:

        print(f"\nWith random.seed({current_seed})\n")

        clf_results_dict = {}
        clf_models_dict = {}

        # ? Further feature selection

        # * Converting Dataset Type

        feature_np = selected_feature_np
        label_np = label_df[[current_label]].to_numpy().astype(int).ravel()

        # * Splitting Dataset

        (x_train, x_test, y_train, y_test) = train_test_split(
            feature_np, label_np, test_size=0.2, random_state=current_seed)

        # * Sampling

        x_resampled, y_resampled = smote(x_train, y_train, current_seed)

        # * FOR EACH CLASSIFIER MODEL ------------------------------------------

        for clf in cfg.classifiers:

            print(f"Building {current_label}: {clf['name']}...")

            # * Tuning

            gscv = tuneClassifer(clf['model'], x_resampled,
                                 y_resampled, clf['param'], cfg.CV, cfg.CV_SCORING)

            best_estimator = gscv.best_estimator_

            # * Training & Testing

            best_estimator.fit(x_resampled, y_resampled)

            scores = get_scoring(best_estimator, x_test, y_test)
            # print(round_scores(scores, 3))

            # * Export results

            clf_results_dict[clf['name']] = scores
            clf_models_dict[clf['name']] = best_estimator

            # * Plotting

            # if clf['name'] == 'Decision Tree':

            #     best_max_leaf_nodes = grid.best_params_['max_leaf_nodes']
            #     plotDecisionTree(best_estimator, feature_names,
            #                      current_label, best_max_leaf_nuodes, SEED)

        label_results_dict.update({current_seed: clf_results_dict})
        label_models_dict.update({current_seed: clf_models_dict})

    # * Save the best models

    label_best_results_dict, label_best_models_dict = save_best_models(
        label_results_dict, label_models_dict)

    print(label_best_results_dict)

    # * Display as Latex tables

    # output_latex_tables[current_label], output_md_tables[current_label] = latextab_per_label(
    #     output_results_dict[current_label], current_label)

    output_all_results_dict.update({current_label: label_results_dict})
    output_best_results_dict.update({current_label: label_best_results_dict})
    output_best_models_dict.update({current_label: label_best_models_dict})

# * Export Models and Results

models_dump(output_best_models_dict)
results_dump(output_all_results_dict)  # TODO
# tables_dump(output_latex_tables, output_md_tables)

print("\n\033[92mDone!\033[0m\n")
