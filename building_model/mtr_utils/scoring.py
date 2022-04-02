from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_scoring(estimator, x_test, y_test):

    f1sc = f1_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test)
    )
    f1scw = f1_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test),
        average='weighted'

    )
    accuracy = accuracy_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test)
    )
    precision = precision_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test),
        zero_division=0
    )
    recall = recall_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test),
        zero_division=0
    )

    scores = {
        'f1-sc': f1sc,
        'f1-scw': f1scw,
        'accura': accuracy,
        'precis': precision,
        'recall': recall,
    }

    return scores


def round_scores(scores, dp):
    for key, value in scores.items():
        scores[key] = round(value, dp)
    return scores
