

def extractLabelDataset(label_df, selected_columns):
    # print(selected_columns)

    # Filter for only recognized samples in labels_df
    rec_label_df = label_df[label_df.recognizable == 1]

    return rec_label_df[['sample'] + selected_columns]
