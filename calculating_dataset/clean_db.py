import numpy as np
import pandas as pd
import re

# Access song_theme_database db
song_theme_feature_database_path = 'data/features/song_theme_feature_database.csv'
features_df = pd.read_csv(song_theme_feature_database_path, na_values=[' NaN'])

# Clean ids
features_df.rename(columns={'Unnamed: 0': 'sample'}, inplace=True)
features_df.iloc[:, 0] = features_df.iloc[:, 0].apply(
    lambda x: re.sub(r'(data\/bin\/.+?(?:\/))(.*)', r'\2', x))

# Replace NaN with 0
features_df.fillna(0, inplace=True)

features_df.to_csv(song_theme_feature_database_path, index=False)
