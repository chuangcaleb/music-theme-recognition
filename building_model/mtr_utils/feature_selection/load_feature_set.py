""" Loads the list of manually shortlisted features """

from typing import Set
import pandas as pd
import json

JSON_PATH = 'building_model/mtr_utils/feature_selection/feature_set.json'
FT_DEF_PATH = 'data/features/song_theme_feature_definitions.csv'

ft_set_dict = json.loads(open(JSON_PATH).read())
ft_def_df = pd.read_csv('data/features/song_theme_feature_definitions.csv')


def explode_bins(feature_str: str) -> Set[Set[str]]:
    feature_bin_count = ft_def_df.loc[ft_def_df.id ==
                                      feature_str]['parallel_dimensions'].values[0]
    if feature_bin_count > 1:
        return {feature_str + '_' + str(x) for x in range(feature_bin_count)}
    else:
        return {feature_str}


def flatten(lst: str) -> Set[str]:
    return {item for sublist in lst for item in sublist}


preselected_feature_set = flatten([
    explode_bins(feature)

    for category in ft_set_dict.values()
    if category['category_enabled'] is True

    for feature, value in category['features'].items()
    if value['feature_enabled'] is True
])
