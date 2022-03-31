from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_scoring(estimator, x_test, y_test):

    f1 = f1_score(
        y_true=y_test,
        y_pred=estimator.predict(x_test)
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
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    }

    return scores


def round_scores(scores, dp):
    for key, value in scores.items():
        scores[key] = round(value, dp)
    return scores
