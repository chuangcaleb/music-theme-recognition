import pandas as pd
from mtr_utils import config as cfg

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

    # Take only the data, discard the id
    data_feature_df = raw_feature_df.drop('id', axis=1)

    # Extract recognizable data from label dataset
    recognz_label_df = raw_label_df[raw_label_df.recognizable == 1]

    # Discard columns that are not selected
    extracted_label_df = recognz_label_df[['id'] + cfg.SELECTED_LABELS]

    # Take only the data, discard the id
    data_label_df = extracted_label_df.drop('id', axis=1)

    # Convert to numpy
    feature_np = data_feature_df.to_numpy()
    label_np = data_label_df.to_numpy()

    print('\nSucessfully imported music theme recognition dataset.')
