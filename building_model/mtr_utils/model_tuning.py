from sklearn.model_selection import GridSearchCV


def get_tuned_classifier(classifier, feature_np, label_np, param_grid, cv, scoring):
    """ Returns a tuned model-agnostic classifier """

    gscv = GridSearchCV(
        classifier,
        param_grid=param_grid,
        refit=True,
        cv=cv,
        scoring=scoring,
        error_score='raise',
        verbose=True
    )

    # Fit with entire dataset
    gscv.fit(feature_np, label_np)

    return gscv.best_estimator_
