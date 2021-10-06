def aggregate_col_values_to_comma_list(col):
    """
        Aggregate all column values in a comma separated list
    """
    return ','.join(set(col.astype(str).values.tolist()))