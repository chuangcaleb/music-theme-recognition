

from sklearn.model_selection import GridSearchCV


def tuneClassifer(classifier, feature_np, label_np, param_grid, cv, score, rand_state):
    """ Tunes a model-agnostic classifier """

    gscv = GridSearchCV(
        classifier,
        param_grid=param_grid,
        refit=True,
        cv=cv,
        scoring=score,
        error_score='raise'
    )
    # Fit with entire dataset
    gscv.fit(feature_np, label_np)

    return gscv
