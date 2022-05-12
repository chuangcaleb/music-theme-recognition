import pandas as pd
from mtr_utils.feature_selection.auto_feature_selection import filter_var_thresh
from mtr_utils import config as cfg
from mtr_utils.feature_selection import load_feature_set

FEATURE_DB_PATH = 'data/features/song_theme_feature_database.csv'
LABEL_DB_PATH = 'data/labels/song_theme_label_database.xlsx'


try:
    # Access song_theme_feature_database
    raw_feature_df = pd.read_csv(FEATURE_DB_PATH)

    # Access song_theme_labels_database
    raw_label_df = pd.read_excel(LABEL_DB_PATH)

except Exception as e:

    print('\nThere was an error in importing the datasets.')
    print(e)

else:

    # * Label

    # Extract recognizable data from label dataset
    recognz_label_df = raw_label_df[raw_label_df.recognizable == 1]

    # Discard labels that are not selected
    extracted_label_df = recognz_label_df[['id'] + cfg.SELECTED_LABELS]

    # Take only the data, discard the id
    data_label_df = extracted_label_df.drop('id', axis=1)

    # * Feature

    # Take only the data, discard the id
    data_feature_df = raw_feature_df.drop('id', axis=1)

    # Load the preselected features
    manual_feature_df = raw_feature_df[load_feature_set.preselected_feature_set]

    selected_features_df, feature_list = filter_var_thresh(
        manual_feature_df, cfg.THRESHOLD_VAL)

    # * Convert to numpy

    feature_np = selected_features_df.to_numpy()
    label_np = data_label_df.to_numpy()

    print('\nSucessfully imported music theme recognition dataset.')
