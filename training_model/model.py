
# *  Import Dataset

from collections import Counter
from sklearn.model_selection import train_test_split
from mtr_utils import config as cfg

from mtr_utils.import_dataset import raw_feature_df, raw_label_df

from mtr_utils.label_dataset_selection import extractLabelDataset

from mtr_utils.feature_selection.load_feature_set import preselected_feature_list
from mtr_utils.feature_selection.auto_feature_selection import filterVarianceThreshold
from mtr_utils.sampling import undersample, oversample, smote


# * Extract data from label dataset

label_df = extractLabelDataset(raw_label_df, cfg.selected_labels)

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_list]

selected_feature_np = filterVarianceThreshold(
    manual_feature_df, cfg.threshold_val)

# * Splitting Dataset

feature_np = selected_feature_np
label_np = label_df[['risk']].to_numpy().astype(int)

(x_train, x_test, y_train, y_test) = train_test_split(
    feature_np, label_np, test_size=0.2, random_state=cfg.rand_state)

# * Sampling

X_resampled, y_resampled = smote(x_train, y_train, cfg.rand_state)
print(sorted(Counter(y_resampled).items()))
