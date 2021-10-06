def aggregate_col_values_to_comma_list(col):
    """
        Aggregate all column values in a comma separated list
    """
    unique_zipcodes  = set(col.astype(str).values.tolist())
    ordered_zipcodes = sorted(list(unique_zipcodes))
    
    return ','.join(ordered_zipcodes)