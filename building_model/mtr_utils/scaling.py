# from mtr_utils import config as cfg
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


# def scale_data(x_train, x_test):

#     scaler = cfg.SCALER

#     scaler.fit(x_train)
#     x_train_scl = scaler.transform(x_train)
#     x_test_scl = scaler.transform(x_test)

#     return scaler, x_train_scl, x_test_scl


class DummyScaler():
    """ Simply returns the same data without scaling """

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        pass

    def __repr__(self):
        return "DummyScaler()"  # TODO: return type(self).__name__ lol


SCALERS = {
    "nrml": MinMaxScaler(),
    "stnd": StandardScaler(),
    "rbst": RobustScaler(),
    # "none": DummyScaler()
}
