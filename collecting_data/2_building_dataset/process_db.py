import pandas as pd


def p_to_1_convert(song_theme_label_database_path):

    label_df = pd.read_excel(song_theme_label_database_path)

    # Replace all 'p' labels with '1'
    label_df.replace('p', 1, inplace=True)

    # Set floats to integers
    # main_df.iloc[:, 2:28] = main_df.iloc[:, 2:28].astype("Int64")

    # Write back to excel
    label_df.to_excel(song_theme_label_database_path, index=False,
                      header=True, freeze_panes=(1, 1))
