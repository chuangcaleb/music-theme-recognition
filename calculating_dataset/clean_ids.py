import pandas as pd
import re

# Access song_theme_database db
song_theme_feature_database_path = 'data/features/song_theme_feature_database.csv'
features_df = pd.read_csv(song_theme_feature_database_path)

features_df.rename(columns={'Unnamed: 0': 'sample'}, inplace=True)
features_df.iloc[:, 0] = features_df.iloc[:, 0].apply(
    lambda x: re.sub(r'(data\/.+?(?:\/))(.*)', r'\2', x))

features_df.to_csv(song_theme_feature_database_path, index=False)
