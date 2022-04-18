from mtr_utils import config as cfg


def scale_data(df):

    scaler = cfg.SCALER

    scaler.fit(df)

    return scaler.transform(df)
