import json
import pickle
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split

from mtr_utils import config as cfg

from mtr_utils.import_dataset import raw_feature_df, raw_label_df

from mtr_utils.label_dataset_selection import extractLabelDataset

from mtr_utils.feature_selection.load_feature_set import preselected_feature_list
from mtr_utils.feature_selection.auto_feature_selection import filterVarianceThreshold

from mtr_utils.sampling import undersample, oversample, smote

from mtr_utils.model_tuning import tuneClassifer

from mtr_utils.scoring import get_scoring, round_scores
from export_results import latextab_per_label

output_models_dict = {}
output_results_dict = {}


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

    print(
        f'\nBuilding model for \033[92m{current_label}\033[0m...')

    output_models_dict[current_label] = {}
    output_results_dict[current_label] = {}

    # ? For loop for current rand_num iteration

    # ? Further feature selection

    # * Converting Dataset

    feature_np = selected_feature_np
    label_np = label_df[[current_label]].to_numpy().astype(int).ravel()

    # * Splitting Dataset

    (x_train, x_test, y_train, y_test) = train_test_split(
        feature_np, label_np, test_size=0.2, random_state=cfg.RAND_STATE)

    # * Sampling

    x_resampled, y_resampled = smote(x_train, y_train, cfg.RAND_STATE)

    # * FOR EACH CLASSIFIER MODEL ----------------------------------------------

    for clf in cfg.classifiers:

        print(
            f"\n{current_label}: {clf['name']}")

        # * Tuning

        gscv = tuneClassifer(clf['model'], feature_np,
                             label_np, clf['param'], cfg.CV, cfg.SCORING)

        best_estimator = gscv.best_estimator_

        # * Training & Testing

        best_estimator.fit(x_resampled, y_resampled)
        best_score = best_estimator.score(x_test, y_test)

        scores = get_scoring(best_estimator, x_test, y_test)
        print(round_scores(scores, 3))

        output_results_dict[current_label][clf['name']] = scores
        output_models_dict[current_label][clf['name']] = best_estimator

        # * Plotting

        # if clf['name'] == 'Decision Tree':

        #     best_max_leaf_nodes = grid.best_params_['max_leaf_nodes']
        #     plotDecisionTree(best_estimator, feature_names,
        #                      current_label, best_max_leaf_nuodes, cfg.RAND_STATE)

        # ? Comparing and printing results

    # * Display as Latex tables

    latextab_per_label(output_results_dict[current_label], current_label)


pickle.dump(
    output_models_dict,
    open("data/output/output_models.pickle", "wb")
)

# for current_label in output_dict:
#     for clf in output_dict[current_label]:
#         output_dict[current_label][clf].pop('model')

with open("data/output/output_results.json", "w") as file:
    json.dump(output_results_dict, file)
output_df = pd.DataFrame.from_dict(output_results_dict)
