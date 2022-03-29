
def extractLabelDataset(label_df, selected_labels_columns):
    """ Extract only relevant data from the label database. 

    1. Select only recognizable samples rows
    2. Select only specified label columns

    """

    # Filter for only recognized samples in labels_df
    rec_label_df = label_df[label_df.recognizable == 1]

    # Only take the selected columns
    # return rec_label_df[['id'] + selected_labels_columns]
    return rec_label_df[selected_labels_columns]
