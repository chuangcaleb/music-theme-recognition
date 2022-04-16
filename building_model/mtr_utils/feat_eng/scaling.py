from sklearn.preprocessing import MinMaxScaler


def normalizeData(df):
    scaler = MinMaxScaler().fit(df)
    return scaler.transform(df)
