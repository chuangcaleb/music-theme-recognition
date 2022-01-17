import os
import pandas as pd

"""
! WARNING: Overwrites the csv! OR: in a non-overwriting way? Compares key index(es), compile & sort unique set, then overwrite
"""

root_path = '../1_scraping_midi/bin'
samples_database_path = './samples_database.csv'
main_df = pd.DataFrame()

# Get list of directories/sources
directory_names = os.listdir(root_path)
# Get list of subfiles
directories_data = [x for x in os.walk(root_path) if x[0] != root_path]

# For each source/directory
for i, directory_data in enumerate(directories_data):

    # Sort this source's samples alphabetically
    directory_data[2].sort()

    # Create a partial dataframe
    current_df = pd.DataFrame(
        {
            "source": directory_names[i],
            "sample": directory_data[2]
        }
    )
    # Append to main dataframe
    main_df = main_df.append(current_df, ignore_index=True)

# Display in console
print(main_df)

# Write to output csv file
main_df.to_csv(samples_database_path, index=False)
