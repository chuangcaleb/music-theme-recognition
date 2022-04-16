from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from mtr_utils import config as cfg
from mtr_utils import import_dataset as data
from mtr_utils.export_results import json_dump, pickle_dump, results_table_dump
from mtr_utils.feat_eng.scaling import normalizeData
from mtr_utils.feature_selection import load_feature_set
from mtr_utils.feature_selection.auto_feature_selection import \
    filterVarianceThreshold
from mtr_utils.label_dataset_selection import extractLabelDataset
from mtr_utils.model_tuning import getTunedClassifer
from mtr_utils.process_results import average_results, save_best_models
from mtr_utils.sampling import oversample, smote, undersample
from mtr_utils.scoring import get_scoring

output_all_results_dict = {}
output_best_results_dict = {}
output_best_models_dict = {}
output_best_params_dict = {}

# * Extract data from label dataset

label_df = extractLabelDataset(data.raw_label_df, cfg.SELECTED_LABELS)

# * Feature Selection

manual_feature_df = data.raw_feature_df[load_feature_set.preselected_feature_list]

selected_features_df, feature_list = filterVarianceThreshold(
    manual_feature_df, cfg.THRESHOLD_VAL)

# * Feature Scaling (Normalization)

scaled_feature_df = normalizeData(selected_features_df)

# ? FEATURE ENGINEERING - merging labels?

# * FOR EACH LABEL -------------------------------------------------------------

for current_label in cfg.SELECTED_LABELS:

    print(f'\n\n> Building model for \033[93m{current_label}\033[0m...')

    label_results_dict = {}
    label_models_dict = {}

    # * FOR EACH RAND_SEED -----------------------------------------------------

    for current_seed in cfg.RAND_SEEDS_LIST:

        print(f"\n{current_label} with random.seed({current_seed}):")

        clf_results_dict = {}
        clf_models_dict = {}

        # ? Further feature selection

        # * Converting Dataset Type

        feature_df = scaled_feature_df
        label_np = label_df[[current_label]].to_numpy().astype(int).ravel()

        # * Splitting Dataset

        (x_train, x_test, y_train, y_test) = train_test_split(
            feature_df, label_np, test_size=cfg.TEST_SIZE, stratify=label_np, random_state=current_seed)

        # * Sampling

        x_resampled, y_resampled = smote(x_train, y_train, current_seed)

        # * FOR EACH CLASSIFIER MODEL ------------------------------------------

        for clf in cfg.classifiers:

            print(f"{clf['name']}...")

            # * Tuning

            best_estimator = getTunedClassifer(clf['model'], x_resampled,
                                               y_resampled, clf['param'], cfg.CV, cfg.BEST_CV_SCORING)

            # * Training

            best_estimator.fit(x_resampled, y_resampled)

            # * Testing

            scores = get_scoring(best_estimator, x_test, y_test)

            # * Export results per classifier

            # Save performance scores
            clf_results_dict[clf['name']] = scores
            # Save model objects UNLESS is a DummyClassifier
            if type(clf['model']) is not DummyClassifier:
                clf_models_dict[clf['name']] = best_estimator

        # * Update results for classifiers per seed

        label_results_dict.update({current_seed: clf_results_dict})
        label_models_dict.update({current_seed: clf_models_dict})

    # * Save only the best seeded models and results for the current label

    label_best_results_dict, label_best_models_dict, label_best_params_dict = save_best_models(
        label_results_dict, label_models_dict)

    # * Update results per current label

    output_all_results_dict.update({current_label: label_results_dict})
    output_best_results_dict.update({current_label: label_best_results_dict})
    output_best_models_dict.update({current_label: label_best_models_dict})
    output_best_params_dict.update({current_label: label_best_params_dict})

# * Get average results

output_avg_results_dict = average_results(output_all_results_dict)

# * Export models and results

json_dump(feature_list, 'final_feature_list')

pickle_dump(output_best_models_dict, 'output_best_models')

json_dump(output_all_results_dict, 'output_all_results', 'results/')
json_dump(output_avg_results_dict, 'output_avg_results', 'results/')
json_dump(output_best_results_dict, 'output_best_results', 'results/')
json_dump(output_best_params_dict, 'output_best_params')

results_table_dump(output_avg_results_dict, 'avg', 'Average')
results_table_dump(output_best_results_dict, 'best', 'Best')

# * Finish!

print("\n\033[92mDone!\033[0m\n")
