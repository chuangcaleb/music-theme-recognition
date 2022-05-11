
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from mtr_utils.config import CLASSIFIERS
from mtr_utils.sampling import smote

from mtr_utils.feature_selection.auto_feature_selection import filter_var_thresh
import mtr_utils.config as cfg

from mtr_utils import import_dataset as data


feature_np = data.feature_np
# label_np = data.label_np
label_np = data.label_np[:, 13]


(
    x_train, x_test, y_train, y_test
) = train_test_split(
    feature_np,
    label_np,
    stratify=label_np,
    random_state=0
)

# Sample
x_train_smp, y_train_smp = smote(x_train, y_train, cfg.RAND_SEED)


# * Pipeline

pipelines = {

    clf['name']: Pipeline(
        [
            ('scaler', None),
            (clf['code'], clf['model'])
        ],
        verbose=True
    )

    for clf in CLASSIFIERS
}


# * Grid Search

parameters = {

    clf['name']:
    {  # Model-agnostic parameters
        'scaler': list(cfg.SCALERS.values()),
    } | {  # Parameters specific to certain models,
        f"{clf['code']}__{k}": v
        for k, v in clf['param'].items()
    }

    for clf in CLASSIFIERS
}


gscv_list = {}
for clf in CLASSIFIERS:

    model_name = clf['name']
    pipe = pipelines[model_name]
    grid = parameters[model_name]

    gscv_list[pipe] = GridSearchCV(pipe, grid, cv=cfg.CV,
                                   verbose=True, scoring=cfg.BEST_CV_SCORING)
    gscv_list[pipe].fit(x_train_smp, y_train_smp)

    print()
    print(model_name)
    print(f'Best F1-score: {gscv_list[pipe].best_score_:.3f}\n')
    print(f'Best parameter set: {gscv_list[pipe].best_params_}\n')
    # print(
    #     f'Scores: {classification_report(y_test, gscv_list[pipe].predict(x_test))}')
