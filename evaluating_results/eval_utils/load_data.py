import json
import pickle
from eval_utils import config as cfg


def load_pickle(path):
    return pickle.load(open(cfg.REL_PATH + path, "rb"))


def load_json(path):
    return json.load(open(cfg.REL_PATH + path, "r"))


models_dict = load_pickle("/output_best_models.pickle")
feature_list = load_json("/final_feature_list.json")
results_dict = load_json("/results/output_best_results.json")
