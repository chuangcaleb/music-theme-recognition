import pandas as pd
import pickle


def config_write(string):
    config_file.write(string + b'\n')


# * Other paths ----------------------------------------------------------------

# Input midi bin's root path
bin_root_path = 'collecting_data/1_scraping_midi/bin/'

# Output path
feat_output_path = 'calculating_dataset/data/song_theme_features_database.xml'
def_output_path = 'calculating_dataset/data/song_theme_definitions_database.xml'

# Features list
all_midi_features_pkl = open('calculating_dataset/all_midi_features.pkl', 'rb')
all_midi_features_list = pickle.load(all_midi_features_pkl)
all_midi_features_pkl.close()

# Access our custom config file
config_file = open('calculating_dataset/themeConfigFile.txt', 'wb')


# * Database -------------------------------------------------------------------

# Access song_theme_database db
song_theme_database_path = 'collecting_data/2_building_dataset/song_theme_database.xlsx'
main_df = pd.read_excel(song_theme_database_path)

# Get recognizable midi paths from database
paths_recognizable_df = main_df[main_df.recognizable == 1].iloc[:, 0:2]
# Generate a Series of all recognizable midi paths
paths_list = bin_root_path + paths_recognizable_df['source'] + \
    '/' + paths_recognizable_df['sample']

# Concatenate Series of strings into one string object
paths_list_string = '\n'.join(paths_list)
# print('\n'.join(paths_list))
# print(*paths_list, sep='\n')


# * Writing --------------------------------------------------------------------

# Options
config_write(b'<jSymbolic_options>')
config_write(b'window_size=0.0')
config_write(b'window_overlap=0.0')
config_write(b'save_features_for_each_window=false')
config_write(b'save_overall_recording_features=true')
config_write(b'convert_to_arff=false')
config_write(b'convert_to_csv=true')

# Features to Extract
config_write(b'<features_to_extract>')
for feature in all_midi_features_list:
    config_write(feature.encode('utf-8'))

# Input Files
config_write(b'<input_files>')
config_write(paths_list_string.encode('utf-8'))

# Output Files
config_write(b'<output_files>')
config_write(b'feature_values_save_path=' +
             feat_output_path.encode('utf-8'))
config_write(b'feature_definitions_save_path=' +
             def_output_path.encode('utf-8'))

config_file.close()
