from sklearn.model_selection import train_test_split

from mtr_utils import config as cfg
from mtr_utils.export_results import (
    pickle_dump, json_dump, results_table_dump)
from mtr_utils.feature_selection.auto_feature_selection import \
    filterVarianceThreshold
from mtr_utils.feature_selection.load_feature_set import \
    preselected_feature_list
from mtr_utils.import_dataset import raw_feature_df, raw_label_df
from mtr_utils.label_dataset_selection import extractLabelDataset
from mtr_utils.model_tuning import tuneClassifer
from mtr_utils.sampling import oversample, smote, undersample
from mtr_utils.process_results import save_best_models, average_results
from mtr_utils.scoring import get_scoring, round_scores

output_best_models_dict = {}
output_results_dict = {}
output_best_results_dict = {}

# * Extract data from label dataset

label_df = extractLabelDataset(raw_label_df, cfg.SELECTED_LABELS)

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_list]

selected_features_df, feature_names = filterVarianceThreshold(
    manual_feature_df, cfg.THRESHOLD_VAL)

# ? FEATURE ENGINEERING - merging labels?

# * FOR EACH LABEL -------------------------------------------------------------

for current_label in cfg.SELECTED_LABELS:

    print(f'\n\n> Building model for \033[93m{current_label}\033[0m...')

    label_results_dict = {}
    label_models_dict = {}

    # * FOR EACH RAND_SEED -----------------------------------------------------

    for current_seed in cfg.RAND_SEEDS_LIST:

        print(f"\nWith random.seed({current_seed})\n")

        clf_results_dict = {}
        clf_models_dict = {}

        # ? Further feature selection

        # * Converting Dataset Type

        feature_df = selected_features_df
        label_np = label_df[[current_label]].to_numpy().astype(int).ravel()

        # * Splitting Dataset

        (x_train, x_test, y_train, y_test) = train_test_split(
            feature_df, label_np, test_size=0.2, random_state=current_seed)

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

            # * Export results per classifier

            clf_results_dict[clf['name']] = scores
            clf_models_dict[clf['name']] = best_estimator

        # * Update results for classifiers per seed

        label_results_dict.update({current_seed: clf_results_dict})
        label_models_dict.update({current_seed: clf_models_dict})

    # * Save only the best seeded models and results for the current label

    label_best_results_dict, label_best_models_dict = save_best_models(
        label_results_dict, label_models_dict)

    # * Update results per current label

    output_results_dict.update({current_label: label_results_dict})
    output_best_results_dict.update({current_label: label_best_results_dict})
    output_best_models_dict.update({current_label: label_best_models_dict})

# * Get average results

output_avg_results_dict = average_results(output_results_dict)

# * Export models and results

json_dump(feature_names, 'final_feature_names')

pickle_dump(output_best_models_dict, 'output_best_models')

json_dump(output_results_dict, 'output_all_results')
json_dump(output_best_results_dict, 'output_best_results')
json_dump(output_avg_results_dict, 'output_avg_results')

results_table_dump(output_best_results_dict, 'best')
results_table_dump(output_avg_results_dict, 'avg')

# * Finish!

print("\n\033[92mDone!\033[0m\n")
