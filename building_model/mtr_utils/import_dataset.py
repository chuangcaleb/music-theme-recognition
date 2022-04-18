import pandas as pd
from mtr_utils import config as cfg

song_theme_feature_database_path = 'data/features/song_theme_feature_database.csv'
song_theme_label_database_path = 'data/labels/song_theme_label_database.xlsx'

try:

    # Access song_theme_feature_database
    raw_feature_df = pd.read_csv(song_theme_feature_database_path)

    # Access song_theme_labels_database
    raw_label_df = pd.read_excel(song_theme_label_database_path)

    # Extract recognizable data from label dataset
    recognz_label_df = raw_label_df[raw_label_df.recognizable == 1]

    # Discard columns that are not selected
    extracted_label_df = recognz_label_df[['id'] + cfg.SELECTED_LABELS]

    # Take only the data, discard the index
    data_label_df = extracted_label_df.drop('id', axis=1)

    print('\nSucessfully imported music theme recognition dataset.')

except:

    print('\nThere was an error in importing the datasets.')
