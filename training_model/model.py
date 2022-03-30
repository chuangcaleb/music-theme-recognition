from sklearn.model_selection import train_test_split

from mtr_utils import config as cfg

from mtr_utils.import_dataset import raw_feature_df, raw_label_df


from mtr_utils.label_dataset_selection import extractLabelDataset

from mtr_utils.feature_selection.load_feature_set import preselected_feature_list
from mtr_utils.feature_selection.auto_feature_selection import filterVarianceThreshold

from mtr_utils.sampling import undersample, oversample, smote

from mtr_utils.model_tuning import tuneClassifer


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

    print(f'\nBuilding model for {current_label}...')

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
    # print(sorted(Counter(y_resampled).items()))

    # * FOR EACH CLASSIFIER MODEL ----------------------------------------------

    for clf in cfg.classifiers:

        print(f"\n{current_label}: {clf['name']}")

        # * Tuning

        grid = tuneClassifer(clf['model'],
                             feature_np, label_np, clf['param'], cfg.CV, cfg.SCORE, cfg.RAND_STATE)

        best_estimator = grid.best_estimator_

        # * Training

        best_estimator.fit(x_resampled, y_resampled)
        best_score = best_estimator.score(x_test, y_test)
        print(f"F1-Score: {best_score}")
        print(grid.best_params_)

        # * Plotting

        # if clf['name'] == 'Decision Tree':

        #     best_max_leaf_nodes = grid.best_params_['max_leaf_nodes']
        #     plotDecisionTree(best_estimator, feature_names,
        #                      current_label, best_max_leaf_nodes, cfg.RAND_STATE)

        # ? Comparing and printing results
