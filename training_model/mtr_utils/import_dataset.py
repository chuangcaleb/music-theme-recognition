import pandas as pd
import os

song_theme_feature_database_path = 'data/features/song_theme_feature_database.csv'
song_theme_label_database_path = 'data/song_theme_label_database.xlsx'

# print(os.getcwd())

try:
    # Access song_theme_feature_database
    raw_feature_df = pd.read_csv(song_theme_feature_database_path)

    # Access song_theme_labels_database
    raw_label_df = pd.read_excel(song_theme_label_database_path)

    print('Sucessfully imported music theme recognition dataset.')

except:

    print('There was an error in importing the datasets.')
