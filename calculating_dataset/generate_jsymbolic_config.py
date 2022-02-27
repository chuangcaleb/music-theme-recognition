import pandas as pd

# * Other paths ----------------------------------------------------------------

# Our midi bin's root path
bin_root_path = 'collecting_data/1_scraping_midi/bin/'

# Output path
output_path = 'calculating_dataset/'

# Access our custom config file
config_file = open("calculating_dataset/themeConfigFile.txt", "wb")


# * Database -------------------------------------------------------------------

# Access song_theme_database db
song_theme_database_path = 'collecting_data/2_building_dataset/song_theme_database.xlsx'
main_df = pd.read_excel(song_theme_database_path)

paths_recognizable_df = main_df[main_df.recognizable == 1].iloc[:, 0:2]

# Generate a Series of all recognizable midi paths
paths_list = paths_recognizable_df['source'] + \
    '/' + paths_recognizable_df['sample']

# Concatenate Series of strings into one string object
paths_list_string = '\n'.join(paths_list)
# print('\n'.join(paths_list))
# print(*paths_list, sep='\n')


# * Writing --------------------------------------------------------------------

# Options
config_file.write(b'<jSymbolic_options>\n')
config_file.write(b'convert_to_arff=False\n')
config_file.write(b'convert_to_csv=True\n')

# Input Files
config_file.write(b"<input_files>\n")
config_file.write(paths_list_string.encode('utf-8') + b'\n')

config_file.write(b"<output_files>\n")
config_file.write(output_path.encode('utf-8'))

config_file.close()
