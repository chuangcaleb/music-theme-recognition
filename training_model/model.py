
# *  Import Dataset

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mtr_utils import config as cfg

from mtr_utils.import_dataset import raw_feature_df, raw_label_df

from mtr_utils.label_dataset_selection import extractLabelDataset

from mtr_utils.feature_selection.load_feature_set import preselected_feature_list
from mtr_utils.feature_selection.auto_feature_selection import filterVarianceThreshold

from mtr_utils.sampling import undersample, oversample, smote

from mtr_utils.model_tuning import tuneClassifer

# * Extract data from label dataset

label_df = extractLabelDataset(raw_label_df, cfg.selected_labels)

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_list]

selected_feature_np = filterVarianceThreshold(
    manual_feature_df, cfg.threshold_val)


# * Iterate for each label

for label in cfg.selected_labels:

    print(f'\nBuilding model for {label}...')

    # * Converting Dataset

    feature_np = selected_feature_np
    label_np = label_df[[label]].to_numpy().astype(int)

    # * Splitting Dataset

    (x_train, x_test, y_train, y_test) = train_test_split(
        feature_np, label_np, test_size=0.2, random_state=cfg.rand_state)

    # * Sampling

    x_resampled, y_resampled = smote(x_train, y_train, cfg.rand_state)
    # print(sorted(Counter(y_resampled).items()))

    # * Tuning

    dt_classifier = DecisionTreeClassifier(random_state=cfg.rand_state)
    dt_gscv = tuneClassifer(dt_classifier,
                            feature_np, label_np, cfg.dt_parameters, cfg.cv, cfg.score, cfg.rand_state)

    best_estimator = dt_gscv.best_estimator_
    best_max_leaf_nodes = dt_gscv.best_params_['max_leaf_nodes']

    # * Training

    best_estimator.fit(x_resampled, y_resampled)
    best_score = best_estimator.score(x_test, y_test)
    print(f"F1-Score for {best_score}")
