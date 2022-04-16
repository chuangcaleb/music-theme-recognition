import json
import pickle
import os
from eval_utils import config as cfg


# ----------------------------------- LOAD ----------------------------------- #

def load_pickle(path):
    return pickle.load(open(cfg.REL_PATH + path, "rb"))


def load_json(path):
    return json.load(open(cfg.REL_PATH + path, "r"))


models_dict = load_pickle("/best_models.pickle")
feature_list = load_json("/feature_list.json")
best_results_dict = load_json("/results/best_results.json")
all_results_dict = load_json("/results/all_results.json")


# ----------------------------------- DUMP ----------------------------------- #

def pickle_dump(dict, filename):
    pickle.dump(dict, open(cfg.OUTPUT_PATH + filename + ".pickle", "wb"))


def json_dump(dict, filename, subdir=''):
    filepath = cfg.OUTPUT_PATH + subdir + filename + ".json"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    json.dump(dict, open(filepath, "w"))
