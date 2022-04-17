from mtr_utils import config as cfg


def scaleData(df):

    scaler = cfg.SCALER

    scaler.fit(df)

    return scaler.transform(df)
