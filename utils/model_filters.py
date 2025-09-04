import pandas as pd

def filter_for_prophet(train_df, valid_df, test_df, cutoff="2025-06-15"):
    # keep only groups whose last training date >= cutoff
    valid_groups = (
        train_df.groupby(["Product_Type", "Market"])["Arrival_Date"]
        .max()
        .reset_index()
    )
    valid_groups = valid_groups[valid_groups["Arrival_Date"] >= pd.to_datetime(cutoff)]
    
    keys = ["Product_Type", "Market"]
    train_df = train_df.merge(valid_groups[keys], on=keys, how="inner")
    valid_df = valid_df.merge(valid_groups[keys], on=keys, how="inner")
    test_df  = test_df.merge(valid_groups[keys], on=keys, how="inner")
    
    return train_df, valid_df, test_df