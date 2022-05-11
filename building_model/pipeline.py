from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from mtr_utils.sampling import smote

from mtr_utils.feature_selection.auto_feature_selection import filter_var_thresh
from mtr_utils.feature_selection import load_feature_set
import mtr_utils.config as cfg

from mtr_utils import import_dataset as data
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

feature_np = data.feature_np
# label_np = data.label_np
label_np = data.label_np[:, 13]

# (
#     x_train, x_test, y_train, y_test
# ) = iterative_train_test_split(
#     feature_np,
#     label_np,
#     test_size=0.2
#     # stratify=label_np,
#     # random_state=0
# )
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

# Pipeline
pipe = Pipeline(
    [
        ('scaler', StandardScaler()),
        ('svc', SVC())

    ],
    verbose=True
)

# pipe.fit(x_train_smp, y_train_smp)

# Binary Relevance

# br_clf = BinaryRelevance(
#     classifier=pipe,
#     require_dense=[False, True]
# )


# Grid Search

parameters = {
    'scaler': [MinMaxScaler(), StandardScaler()],
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    'svc__C': [00.01, 0.1, 1, 10, 100]
}

gscv = GridSearchCV(pipe, parameters, cv=cfg.CV,
                    verbose=True, scoring=cfg.BEST_CV_SCORING)
gscv.fit(x_train_smp, y_train_smp)


print(f'Best F1-score: {gscv.best_score_:.3f}\n')
print(f'Best parameter set: {gscv.best_params_}\n')
print(f'Scores: {classification_report(y_test, gscv.predict(x_test))}')

# manual_feature_df = raw_feature_df[load_feature_set.preselected_feature_set]
# # Automatic selection
# selected_features_df, feature_list = filter_var_thresh(
#     manual_feature_df, 0)

# # # * Splitting Dataset

# # x_train, x_test, y_train, y_test = train_test_split(
# #     feature_np, label_np, test_size=cfg.TEST_SIZE, stratify=label_np, random_state=current_seed)

# # * Sampling

# x_train_smp, y_train_smp = smote(selected_features_df, raw_label_df, 0)

# # * Feature Scaling

# scaler, x_train_scl, x_test_scl = scale_data(x_train_smp, x_train_smp)

# parameters = [
#     {
#         'classifier': [MultinomialNB()],
#         'classifier__alpha': [0.7, 1.0],
#     },
#     {
#         'classifier': [SVC()],
#         'classifier__kernel': ['rbf', 'linear'],
#     },
# ]

# clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
# clf.fit(x_train_scl.values, raw_label_df)

# print(clf.best_params_, clf.best_score_)
