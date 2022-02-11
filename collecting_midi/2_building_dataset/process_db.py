import pandas as pd


def convert(song_theme_database_path):

    main_df = pd.read_excel(song_theme_database_path)

    # Replace all 'p' labels with '1'
    main_df.replace('p', 1, inplace=True)

    # Set floats to integers
    # main_df.iloc[:, 2:28] = main_df.iloc[:, 2:28].astype("Int64")

    # Write back to excel
    main_df.to_excel(song_theme_database_path, index=False,
                     header=True, freeze_panes=(1, 1))


# song_theme_database_path = './song_theme_database.xlsx'
# convert(song_theme_database_path)
