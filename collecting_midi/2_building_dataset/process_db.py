import pandas as pd


song_theme_database_path = './song_theme_database.xlsx'
main_df = pd.read_excel(song_theme_database_path)

# Replace all 'p' labels with '1'
main_df.replace('p', 1, inplace=True)

#
main_df.style.set_properties(subset=['sample'], **{'width': '300px'})

# Write back to excel
main_df.to_excel(song_theme_database_path, index=False,
                 header=True, freeze_panes=(1, 1))
