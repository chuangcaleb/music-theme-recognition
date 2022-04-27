import warnings
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

from mtr_utils import config as cfg
from mtr_utils import import_dataset as data
from mtr_utils.export_results import (export_config, json_dump, pickle_dump,
                                      results_table_dump)
from mtr_utils.feature_selection import load_feature_set
from mtr_utils.feature_selection.auto_feature_selection import \
    filter_var_thresh
from mtr_utils.model_tuning import get_tuned_classifier
from mtr_utils.sampling import oversample, smote, undersample
from mtr_utils.save_best import save_best_models
from mtr_utils.scaling import scale_data
from mtr_utils.scoring import get_scoring

# warnings.simplefilter("ignore", category=ConvergenceWarning)

output_all_results_dict = {}
output_best_results_dict = {}
output_best_models_dict = {}
output_best_params_dict = {}

raw_feature_df = data.raw_feature_df
raw_label_df = data.data_label_df

# * Feature Selection

# Manual selection
manual_feature_df = raw_feature_df[load_feature_set.preselected_feature_set]

# Automatic selection
selected_features_df, feature_list = filter_var_thresh(
    manual_feature_df, cfg.THRESHOLD_VAL)

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

        # * Converting Dataset Type

        feature_np = selected_features_df.values
        label_np = raw_label_df[[current_label]].to_numpy().astype(int).ravel()

        # * Splitting Dataset

        x_train, x_test, y_train, y_test = train_test_split(
            feature_np, label_np, test_size=cfg.TEST_SIZE, stratify=label_np, random_state=current_seed)

        # * Sampling

        x_train_smp, y_train_smp = smote(x_train, y_train, current_seed)

        # * Feature Scaling

        scaler, x_train_scl, x_test_scl = scale_data(x_train_smp, x_test)

        # * FOR EACH CLASSIFIER MODEL ------------------------------------------

        for clf in cfg.CLASSIFIERS:

            print(f"{clf['name']}...")

            # * Tuning

            best_estimator = get_tuned_classifier(clf['model'], x_train_scl,
                                                  y_train_smp, clf['param'], cfg.CV, cfg.BEST_CV_SCORING)

            # * Training

            best_estimator.fit(x_train_scl, y_train_smp)

            # * Testing

            scores = get_scoring(best_estimator, x_test_scl, y_test)

            # * Export results per classifier

            # Save performance scores
            clf_results_dict[clf['name']] = scores
            # Save model objects IF is not a baseline classifier
            if clf['name'] in cfg.ACTUAL_CLASSIFIERS:
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


# * Export data for this run

json_dump(feature_list, 'final_feature_list')

json_dump(output_all_results_dict, 'results_all', 'results/')
json_dump(output_best_results_dict, 'results_best', 'results/')
json_dump(output_best_params_dict, 'params_best')

pickle_dump(output_best_models_dict, 'models_best')

json_dump(export_config(), 'run_config')

results_table_dump(output_best_results_dict, 'results_best', 'Best')

# * Finish!

print("\n\033[92mDone!\033[0m\n")
