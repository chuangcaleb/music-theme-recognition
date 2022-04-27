import os.path
import pandas as pd

"""
Compares key index(es), compile & sort unique set, then overwrite?
"""

BIN_DIR = 'data/bin'
LABEL_DB_PATH = 'data/labels/song_theme_label_database.xlsx'

# Get list of directories/sources
directory_names = os.listdir(BIN_DIR)
# Get list of subfiles
directories_data = [x for x in os.walk(BIN_DIR) if x[0] != BIN_DIR]


if not os.path.exists(LABEL_DB_PATH):

    # For each source/directory
    for i, directory_data in enumerate(directories_data):

        print(i, directory_names[i])

        # Sort this source's samples alphabetically
        directory_data[2].sort()

        # Create a partial dataframe
        current_df = pd.DataFrame({
            "id": directory_data[2],
            "source": directory_names[i]
        })

        # Append to main dataframe
        label_df = label_df.append(current_df, ignore_index=True)

    # Display in console
    print(label_df)

    # Write to output csv file
    label_df.to_excel(LABEL_DB_PATH, index=False)

else:

    print('\nLabel database already exists!\n')
