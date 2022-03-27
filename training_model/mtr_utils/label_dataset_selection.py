

def extractLabelDataset(label_df, selected_columns):
    # Filter for only recognized samples in labels_df
    rec_label_df = label_df[label_df.recognizable == 1]
    return rec_label_df[selected_columns]

    # # Filter out unused labels and metadata
    # main_label_df.drop(
    #     columns=main_label_df.columns[19:], axis=1, inplace=True)  # unused labels
    # main_label_df.drop(
    #     columns=main_label_df.columns[0:4], axis=1, inplace=True)  # metadata
