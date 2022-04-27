import pandas as pd

FEAT_ROOT_DIR = 'data/features/'
XML_PATH = FEAT_ROOT_DIR + 'song_theme_feature_definitions.xml'
OUTPUT_PATH = FEAT_ROOT_DIR + 'song_theme_feature_definitions.csv'


def snake_case(string):
    return string.replace(' ', '_')


# Reading data from the xml file
with open(XML_PATH, 'r') as f:
    feature_xml = f.read()

ft_def_df = pd.read_xml(feature_xml, xpath='.//feature')

# Add an extra row, for featue ids
ft_def_df['id'] = ft_def_df['name'].apply(snake_case)

# Export

ft_def_df.to_csv(OUTPUT_PATH)
print('\nSuccessfully dumped feature definitions!\n')
