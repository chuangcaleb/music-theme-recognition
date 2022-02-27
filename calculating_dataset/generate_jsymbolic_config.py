import pandas as pd

# Needs to be in diff path
song_theme_database_path = 'collecting_data/2_building_dataset/song_theme_database.xlsx'
main_df = pd.read_excel(song_theme_database_path)

print(main_df)
