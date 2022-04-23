import pandas as pd
from eval_utils import config as cfg
from eval_utils import load_results as data


FILEPATH = cfg.RUN_DIR + 'results/' + 'flat_results.csv'

all_results_dict = data.all_results_dict

result = [
    [label, seed, clf, metric, score]
    for label, seed_dict in all_results_dict.items()
    for seed, clf_dict in seed_dict.items()
    for clf, scores_dict in clf_dict.items()
    for metric, score in scores_dict.items()
]

columns = ["label", "seed", "clf", "metric", "score"]

all_results_df = pd.DataFrame(result, columns=columns)

all_results_df.to_csv(FILEPATH)
