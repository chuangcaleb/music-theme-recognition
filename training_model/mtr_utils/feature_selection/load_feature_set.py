import json

# Path is relative to training_model instead of root, because the model.ipynb is run relative to itself and not root.
# json_file_path = 'feature_selection/feature_set.json'
# output_file_path = 'feature_selection/manually_preselected_features.txt'
json_file_path = 'training_model/mtr_utils/feature_selection/feature_set.json'
output_file_path = 'training_model/mtr_utils/feature_selection/manually_preselected_features.txt'


def load_presel_features(json_file_path, output_file_path):

    features_file = json.loads(
        open(json_file_path).read()
    )

    preselected_feature_list = []

    # For each category in the file
    for category_name in features_file:

        # Process category only if true, skip otherwise
        if features_file[category_name]['category_enabled'] == True:

            # Get category object
            current_category = features_file[category_name]['features']

            # For each feature in current category
            for feature_name in current_category:

                # Get feature object
                current_feature = current_category[feature_name]

                # If feature is enabled, skip otherwise
                if current_feature['feature_enabled'] == True:

                    # If bin_size exists, create a list of features
                    if 'bin_size' in current_feature:
                        current_feature_range = [
                            feature_name + '_' + str(x) for x in range(current_feature['bin_size'])
                        ]
                        # Append range of features to main list
                        preselected_feature_list.extend(current_feature_range)
                    # Else, just add the single feature
                    else:
                        # Append single feature to main list
                        preselected_feature_list.append(feature_name)

    # Write output to .txt file too
    with open(output_file_path, 'w') as f:
        for feature in preselected_feature_list:
            f.write(feature + '\n')

    # print('\nSuccessfully written manually preselected features to ' + output_file_path + '\n')
    print(
        f'\nSucessfully loaded {len(preselected_feature_list)} manually selected features.')

    return(preselected_feature_list)


preselected_feature_list = load_presel_features(
    json_file_path, output_file_path)
