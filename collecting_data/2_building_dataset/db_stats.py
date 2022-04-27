import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_db import *

LABELS_DIR = 'data/labels/'
LABEL_DB_PATH = LABELS_DIR + 'song_theme_label_database.xlsx'
STATS_EXPORT_PATH = LABELS_DIR + 'label_stats_summary.json'

# Convert all p's to 1's
p_to_1_convert(LABEL_DB_PATH)

# Import data
label_df = pd.read_excel(LABEL_DB_PATH)

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
perc_recognizable = percentage(recognizable_count, total_count)

processed_count = int(label_df.recognizable.count())
perc_processed = percentage(processed_count, total_count)

unprocessed_count = total_count - processed_count

perc_recog_procs = percentage(recognizable_count, processed_count)

# * Count label occurrences

# Drop all un-recognizable samples from the df, from here onwards
label_df = label_df[label_df.recognizable == 1]

# Count label occurrences
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

print(f"Total number of samples: {total_count}")
print(f"Recognizable samples: {recognizable_count} ({perc_recognizable}%)")
print(f"Processed samples: {processed_count} ({perc_processed}%)")
print(f"-> Unprocessed samples: {unprocessed_count}")
print(f"Recognized / Processed samples: ({perc_recog_procs}%)")

print()

print("> Label Statistics")
print(label_stats_df)
print("\n> Sorted Label Statistics")
print(sorted_label_stats_df)

print()

# sorted_label_stats_df.plot(kind='bar', y=1.0, xlabel='Labels',
#                            ylabel='Frequency', legend=False, title='Theme Label Frequencies in Samples Dataset')
# plt.show()


sorted_label_stats_df.plot(
    kind='bar',
    y=1.0,
    xlabel='Labels',
    ylabel='Frequency',
    legend=False,
    title='Theme Label Frequencies in Samples Dataset'
)

plt.tight_layout()
plt.savefig(LABELS_DIR + 'label_freq.png')

# * EXPORT

sorted_label_stats_dict = sorted_label_stats_df.to_dict()

stats_dict = {
    'total_count': total_count,
    'recognized_count': recognizable_count,
    'recognizable_perc': perc_recognizable,
    'processed_samples_count': processed_count,
    'processed_samples_perc': perc_processed,
    'unprocessed_count': unprocessed_count,
    'recog_procs_perc': perc_recog_procs,
} | sorted_label_stats_dict

json.dump(stats_dict, open(STATS_EXPORT_PATH, "w"))

print('Saved label statistics to ' + STATS_EXPORT_PATH)
