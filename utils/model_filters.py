import pandas as pd

def filter_for_prophet(train_df, valid_df, test_df, cutoff="2025-06-15"):
    """
    Filters the train, validation, and test datasets for Prophet modeling.
    
    Only keeps product-market groups that have data up to at least the specified cutoff date.
    This ensures the Prophet model has enough historical data to learn trends before making predictions.

    Parameters:
        train_df (pd.DataFrame): Training dataset.
        valid_df (pd.DataFrame): Validation dataset.
        test_df (pd.DataFrame): Test dataset.
        cutoff (str or datetime): The minimum last date in training data for a product-market group
                                  to be considered. Default is "2025-06-15".
    
    Returns:
        tuple: Filtered (train_df, valid_df, test_df)
    """
    # Identify groups with sufficient data in training set
    valid_groups = (
        train_df.groupby(["Product_Type", "Market"])["Arrival_Date"]
        .max()
        .reset_index()
    )
    valid_groups = valid_groups[valid_groups["Arrival_Date"] >= pd.to_datetime(cutoff)]
    
    keys = ["Product_Type", "Market"]
    
    # Keep only the valid product-market groups in all datasets
    train_df = train_df.merge(valid_groups[keys], on=keys, how="inner")
    valid_df = valid_df.merge(valid_groups[keys], on=keys, how="inner")
    test_df  = test_df.merge(valid_groups[keys], on=keys, how="inner")
    
    return train_df, valid_df, test_df