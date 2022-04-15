from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def get_scoring(estimator, x_test, y_test):

    y_predictions = estimator.predict(x_test)

    f1bin = f1_score(
        y_true=y_test,
        y_pred=y_predictions
    )
    f1mac = f1_score(
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
    )
    rocauc = roc_auc_score(
        y_true=y_test,
        y_score=y_predictions,
    )

    scores = {
        'f1-bin': f1bin,
        'f1-mac': f1mac,
        'accura': accuracy,
        'precis': precision,
        'recall': recall,
        'rocauc': rocauc,
    }

    return scores
