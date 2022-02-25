import os
import pandas as pd

"""
! WARNING: Overwrites the csv! OR: in a non-overwriting way? Compares key index(es), compile & sort unique set, then overwrite
"""

root_path = '../1_scraping_midi/bin'
song_theme_database_path = './song_theme_database.xlsx'
main_df = pd.DataFrame()

# Get list of directories/sources
directory_names = os.listdir(root_path)
# Get list of subfiles
directories_data = [x for x in os.walk(root_path) if x[0] != root_path]

# For each source/directory
for i, directory_data in enumerate(directories_data):

    print(directory_names[i])

    # Sort this source's samples alphabetically
    directory_data[2].sort()

    # Create a partial dataframe
    current_df = pd.DataFrame(
        {
            "sample": directory_data[2],
            "source": directory_names[i]
        }
    )
    # Append to main dataframe
    main_df = main_df.append(current_df, ignore_index=True)

# Display in console
print(main_df)

# Write to output csv file
main_df.to_excel(song_theme_database_path, index=False)