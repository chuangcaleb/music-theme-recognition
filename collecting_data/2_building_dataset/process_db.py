import pandas as pd


def p_to_1_convert(label_db_path):
    """ Converts all occurrences of 'p' to '1' """

    label_df = pd.read_excel(label_db_path)

    # Replace all 'p' labels with '1'
    label_df.replace('p', 1, inplace=True)

    # Set floats to integers
    # main_df.iloc[:, 2:28] = main_df.iloc[:, 2:28].astype("Int64")

    # Write back to excel
    label_df.to_excel(label_db_path, index=False,
                      header=True, freeze_panes=(1, 1))
