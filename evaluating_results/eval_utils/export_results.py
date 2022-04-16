
import json
import os
import pickle

from eval_utils import config as cfg

# * Dump -----------------------------------------------------------------------


def pickle_dump(dict, filename):
    pickle.dump(dict, open(cfg.OUTPUT_PATH + filename + ".pickle", "wb"))


def json_dump(dict, filename, subdir=''):
    filepath = cfg.OUTPUT_PATH + subdir + filename + ".json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    json.dump(dict, open(filepath, "w"))


def txt_dump(output_tables, name, ext):
    """ Helper function to write dictionaries to text files """

    filepath = cfg.OUTPUT_PATH + 'tables/' + name + ext
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w") as f:

        f.write(f'# {cfg.RUN_ID}: {name} results\n')

        for tableId in output_tables:

            f.write('\n## ' + tableId + '\n\n' + output_tables[tableId] + '\n')
