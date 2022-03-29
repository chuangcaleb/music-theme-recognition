import os
import pandas as pd

"""
! WARNING: Overwrites the csv! OR: in a non-overwriting way? Compares key index(es), compile & sort unique set, then overwrite
Nah not enough time. If accidentally overwritten, just restore from last git commit.
"""

root_path = 'data/bin'
song_theme_database_path = 'data/song_theme_label_database.xlsx'
label_df = pd.DataFrame()

# Get list of directories/sources
directory_names = os.listdir(root_path)
# Get list of subfiles
directories_data = [x for x in os.walk(root_path) if x[0] != root_path]

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
label_df.to_excel(song_theme_database_path, index=False)
