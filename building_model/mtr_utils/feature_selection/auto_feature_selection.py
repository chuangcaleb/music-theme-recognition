from sklearn.feature_selection import VarianceThreshold


def filter_var_thresh(df, threshold_val):
    """Removes all columns of a dataframe that fall below a given threshold of 
    variance """

    constant_filter = VarianceThreshold(threshold=threshold_val)
    constant_filter.fit(df)

    constant_columns = [column for column in df.columns
                        if column not in
                        df.columns[constant_filter.get_support()]]
    var_feature_np = constant_filter.transform(df)

    print(f"\nStarted with {df.shape[1]} features.")
    print(
        f"Removed {len(constant_columns)} feature(s) that have a variance of less than {threshold_val}.")

    for column in constant_columns:
        print("Removed:", column)

    print(f"There are {(var_feature_np.shape[1])} features left.")

    variant_features = [x for x in df.columns if not x in constant_columns]

    return df[variant_features], variant_features
