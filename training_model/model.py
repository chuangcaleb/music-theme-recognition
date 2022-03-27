
# *  Import Dataset

from mtr_utils import config as cfg
from mtr_utils.import_dataset import raw_feature_df, raw_label_df
from mtr_utils.feature_selection.load_feature_set import preselected_feature_list
from mtr_utils.feature_selection.auto_feature_selection import filterVarianceThreshold

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_list]

selected_feature_df = filterVarianceThreshold(
    manual_feature_df, cfg.threshold_val)
