import json
import pickle
from eval_utils import config as cfg


# ----------------------------------- LOAD ----------------------------------- #

def load_pickle(path):
    return pickle.load(open(cfg.RUN_DIR + path, "rb"))


def load_json(path):
    return json.load(open(cfg.RUN_DIR + path, "r"))


models_dict = load_pickle("models_best.pickle")
feature_list = load_json("final_feature_list.json")
best_results_dict = load_json("results/results_best.json")
all_results_dict = load_json("results/results_all.json")
config_dict = load_json("run_config.json")
