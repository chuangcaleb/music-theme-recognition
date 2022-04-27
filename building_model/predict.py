
import pickle
import random
from mtr_utils.import_dataset import raw_feature_df, raw_label_df
from mtr_utils.feature_selection.load_feature_set import preselected_feature_set
from mtr_utils.feature_selection.auto_feature_selection import filter_var_thresh
from mtr_utils import config as cfg

pickle_data = pickle.load(open("data/output/output_models.pickle", "rb"))

model = pickle_data['risk']['Naive Bayes']['model']

# * Feature Selection

manual_feature_df = raw_feature_df[preselected_feature_set]

selected_feature_np, feature_list = filter_var_thresh(
    manual_feature_df, cfg.THRESHOLD_VAL)

random_feature = selected_feature_np.iloc[92, ].values.reshape(1, -1)

result = model.predict(random_feature)
print(result)
