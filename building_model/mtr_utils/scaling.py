from mtr_utils import config as cfg


def scale_data(x_train, x_test):

    scaler = cfg.SCALER

    scaler.fit(x_train)
    x_train_smp = scaler.transform(x_train)
    x_test_smp = scaler.transform(x_test)

    return scaler, x_train_smp, x_test_smp


class DummyScaler():
    """ Simply returns the same data without scaling """

    def transform(self, X):
        return X

    def fit(self, X, y=None):
        pass
