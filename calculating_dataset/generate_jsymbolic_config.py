import pandas as pd
from feature_list import ALL_MIDI_FEATURES_SET


def config_write(string):
    config_file.write((string + '\n').encode('utf-8'))


# * Other paths ----------------------------------------------------------------

DATA_ROOT_DIR = 'data/'

# Input midi bin's root dir
BIN_ROOT_DIR = DATA_ROOT_DIR + 'bin/'

LABEL_DB_PATH = DATA_ROOT_DIR + 'labels/song_theme_label_database.xlsx'

# Output path
FEAT_ROOT_DIR = DATA_ROOT_DIR + 'features/'
FEAT_OUTPUT_PATH = FEAT_ROOT_DIR + 'song_theme_feature_database.xml'
DEF_OUTPUT_PATH = FEAT_ROOT_DIR + 'song_theme_feature_definitions.xml'

CONFIG_PATH = FEAT_ROOT_DIR + 'theme_jsymb_config.txt'

# * Import Data ----------------------------------------------------------------

# Access song_theme_label_database db
label_df = pd.read_excel(LABEL_DB_PATH)

# Access our custom config file
config_file = open(CONFIG_PATH, 'wb')

# * Process Data ---------------------------------------------------------------


# Get recognizable midi paths from database
paths_recognizable_df = label_df[label_df.recognizable == 1].iloc[:, 0:2]
# Generate a Series of all recognizable midi paths
paths_list = BIN_ROOT_DIR + paths_recognizable_df['source'] + \
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
for feature in ALL_MIDI_FEATURES_SET:
    config_write(feature)

# Input Files
config_write('<input_files>')
config_write(paths_list_string)

# Output Files
config_write('<output_files>')
config_write('feature_values_save_path=' + FEAT_OUTPUT_PATH)
config_write('feature_definitions_save_path=' + DEF_OUTPUT_PATH)

config_file.close()
print('\nGenerated config file at ' + CONFIG_PATH, end='\n\n')
