import pandas as pd
from process_db import *
import matplotlib.pyplot as plt

song_theme_label_database_path = 'data/song_theme_label_database.xlsx'

# Convert all p's to 1's
p_to_1_convert(song_theme_label_database_path)

label_df = pd.read_excel(song_theme_label_database_path)

# * Aux methods


def percentage(positive, total):
    return round(((positive / total) * 100), 1)


# * Count overall statistics

"""
Recognized/total = (countif recognizable == 1) / total
Processed = (countif recognizable != NaN) / total
Recog/Processed = (countif recognizable == 1) / (countif recognizable != NaN)
 """

total_count = len(label_df.index)

recognizable_count = len(label_df[label_df.recognizable == 1])
perc_recognizable = "(" + \
    str(percentage(recognizable_count, total_count)) + "%)"

processed_count = label_df.recognizable.count()
perc_processed = "(" + \
    str(percentage(processed_count, total_count)) + "%)"

unprocessed_count = total_count - processed_count

perc_recog_procs = "(" + \
    str(percentage(recognizable_count, processed_count)) + "%)"

# * Count label values
label_stats_df = label_df.iloc[:, 4:19].apply(pd.value_counts).T

# Casting as integer
label_stats_df = label_stats_df.astype("Int64")

# Replace NaN with 0
label_stats_df.fillna(0, inplace=True)

# * Calculate label percentages
label_stats_df['%'] = label_stats_df[1.0] / \
    (label_stats_df[0.0] + label_stats_df[1.0])
# Convert to percentage
label_stats_df['%'] = (label_stats_df['%'] * 100).round(1)

sorted_label_stats_df = label_stats_df.sort_values(by='%', ascending=False)

# * PRINT

print("\n\n\033[92mMusic Theme Database Statistics\033[0m \n")

print("Total number of samples:", total_count)
print("Recognizable samples:",
      recognizable_count, perc_recognizable)
print("Processed samples:", processed_count, perc_processed)
print("-> Unprocessed samples:", unprocessed_count)
print("Recognized / Processed samples:", perc_recog_procs)

print()

print("> Label Statistics")
print(label_stats_df)
print("\n> Sorted Label Statistics")
print(sorted_label_stats_df)

print()

sorted_label_stats_df.plot(kind='bar', y=1.0, xlabel='Labels',
                           ylabel='Frequency', legend=False, title='Theme Label Frequencies in Samples Dataset')
plt.show()
