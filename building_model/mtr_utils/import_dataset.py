import pandas as pd
from mtr_utils import config as cfg

song_theme_feature_database_path = 'data/features/song_theme_feature_database.csv'
song_theme_label_database_path = 'data/labels/song_theme_label_database.xlsx'


def extractLabelDataset(label_df, selected_labels_columns):
    """ Extract only relevant data from the label database. 

    1. Select only recognizable samples rows
    2. Select only specified label columns
    """

    # Filter for only recognized samples in labels_df
    rec_label_df = label_df[label_df.recognizable == 1]

    # Only take the selected columns
    return rec_label_df[selected_labels_columns]


try:
    # Access song_theme_feature_database
    raw_feature_df = pd.read_csv(song_theme_feature_database_path)

    # Access song_theme_labels_database
    raw_label_df = pd.read_excel(song_theme_label_database_path)

    # Extract data from label dataset
    extracted_label_df = extractLabelDataset(raw_label_df, cfg.SELECTED_LABELS)

    print('\nSucessfully imported music theme recognition dataset.')

except:

    print('\nThere was an error in importing the datasets.')
