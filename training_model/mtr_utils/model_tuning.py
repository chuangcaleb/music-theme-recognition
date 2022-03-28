

from sklearn.model_selection import GridSearchCV


def tuneClassifer(classifier, feature_np, label_np, dt_parameters, cv, score, rand_state):
    """ Tunes a decision tree model """

    gscv = GridSearchCV(
        classifier,
        param_grid=dt_parameters,
        refit=True,
        cv=cv,
        scoring=score,
        error_score='raise'
    )
    # Fit with entire dataset
    gscv.fit(feature_np, label_np)

    return gscv
