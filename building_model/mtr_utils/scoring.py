from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def get_scoring(estimator, x_test, y_test):

    y_predictions = estimator.predict(x_test)

    f1sc = f1_score(
        y_true=y_test,
        y_pred=y_predictions
    )
    f1scm = f1_score(
        y_true=y_test,
        y_pred=y_predictions,
        average='macro'
    )
    accuracy = accuracy_score(
        y_true=y_test,
        y_pred=y_predictions
    )
    precision = precision_score(
        y_true=y_test,
        y_pred=y_predictions,
        zero_division=0
    )
    recall = recall_score(
        y_true=y_test,
        y_pred=y_predictions,
        zero_division=0
    ),
    # roc_auc = roc_auc_score(
    #     y_true=y_test,
    #     y_pred=y_predictions,
    # )

    scores = {
        'f1-sc': f1sc,
        'f1-scm': f1scm,
        'accura': accuracy,
        'precis': precision,
        'recall': recall,
        # 'roc-auc': roc_auc,
    }

    return scores


def round_scores(scores, dp):
    for key, value in scores.items():
        scores[key] = round(value, dp)
    return scores
