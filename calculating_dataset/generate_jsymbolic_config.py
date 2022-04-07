import pandas as pd
import feature_dump as feature_dump_list


def config_write(string):
    config_file.write((string + '\n').encode('utf-8'))


# * Other paths ----------------------------------------------------------------

# Input midi bin's root path
bin_root_path = 'data/bin/'

# Output path
feat_output_path = 'data/features/song_theme_feature_database.xml'
def_output_path = 'data/features/song_theme_feature_definitions.xml'

# Access our custom config file
config_file = open('calculating_dataset/themeConfigFile.txt', 'wb')


# * Database -------------------------------------------------------------------

# Access song_theme_label_database db
song_theme_label_database_path = 'data/labels/song_theme_label_database.xlsx'
label_df = pd.read_excel(song_theme_label_database_path)

# Get recognizable midi paths from database
paths_recognizable_df = label_df[label_df.recognizable == 1].iloc[:, 0:2]
# Generate a Series of all recognizable midi paths
paths_list = bin_root_path + paths_recognizable_df['source'] + \
    '/' + paths_recognizable_df['id']

# Concatenate Series of strings into one string object
paths_list_string = '\n'.join(paths_list)


# * Writing --------------------------------------------------------------------

# Options
config_write('<jSymbolic_options>')
config_write('window_size=0.0')
config_write('window_overlap=0.0')
config_write('save_features_for_each_window=false')
config_write('save_overall_recording_features=true')
config_write('convert_to_arff=false')
config_write('convert_to_csv=true')

# Features to Extract
config_write('<features_to_extract>')
for feature in feature_dump_list.all_midi_features_list:
    config_write(feature)

# Input Files
config_write('<input_files>')
config_write(paths_list_string)

# Output Files
config_write('<output_files>')
config_write('feature_values_save_path=' + feat_output_path)
config_write('feature_definitions_save_path=' + def_output_path)

config_file.close()
