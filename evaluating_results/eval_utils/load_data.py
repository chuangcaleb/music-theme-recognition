import json
import pickle


def load_pickle(path):
    return pickle.load(open(path, "rb"))

def load_json(path):
    return json.load(open(path, "r"))
