class DataSplitter:
    """
    Utility class to split a commodity price dataset into train, validation, and test sets,
    while filtering out groups with insufficient data and safely dropping helper columns.
    
    Steps performed:
    1. Assign each row to 'train', 'valid', or 'test' based on Arrival_Date.
    2. Filter out product-market groups that do not meet minimum sample thresholds.
    3. Return split datasets with optional leakage/helper columns removed.
    """

    def __init__(self, df, date_dict: dict, thresholds: dict):
        """
        Parameters:
        df (pd.DataFrame): Input dataset containing 'Arrival_Date', 'Product_Type', 'Market', etc.
        date_dict (dict): Dictionary specifying date ranges for splits.
            e.g., {'train_start': '2023-06-01', 'train_end': '2024-06-30', ...}
        thresholds (dict): Minimum number of samples per split to keep a product-market group.
            e.g., {'train': 30, 'valid': 10, 'test': 10}
        """
        self.df = df
        self.date_dict = date_dict
        self.thresholds = thresholds

    def assign_split(self, date):
        """Assign a split label (train/valid/test) based on date ranges."""
        if self.date_dict['train_start'] <= date.strftime("%Y-%m-%d") <= self.date_dict['train_end']:
            return "train"
        elif self.date_dict['valid_start'] <= date.strftime("%Y-%m-%d") <= self.date_dict['valid_end']:
            return "valid"
        else:
            return "test"

    def assign_splits(self):
        """Add 'Split' column to dataframe based on Arrival_Date."""
        self.df['Split'] = self.df['Arrival_Date'].apply(lambda d: self.assign_split(d))

    def filter_groups(self):
        """
        Filter out product-market groups that have insufficient samples in any split.
        Ensures the model only trains on groups with enough historical data.
        """
        group_counts = (
            self.df.groupby(["Product_Type", "Market", "Split"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        group_counts = group_counts[
            ~((group_counts['train'] <= self.thresholds['train']) |
              (group_counts['valid'] <= self.thresholds['valid']) |
              (group_counts['test'] <= self.thresholds['test']))
        ]
        self.df = self.df.merge(group_counts, on=['Product_Type', 'Market'], how='right')

    def split_datasets(self):
        """Return train, validation, and test DataFrames."""
        train_df = self.df[self.df['Split'] == 'train']
        valid_df = self.df[self.df['Split'] == 'valid']
        test_df = self.df[self.df['Split'] == 'test']
        return train_df, valid_df, test_df

    def drop_columns(self, dataframe):
        """
        Drop helper or leakage columns safely if they exist.
        Typically used after splitting to clean datasets before modeling.
        """
        columns_to_drop = ['Split', 'train', 'valid', 'test', 
                           'Modal_Price', 'Max_Price', 'Min_Price',
                           'Modal_Price_filled']
        existing_cols = [col for col in columns_to_drop if col in dataframe.columns]
        return dataframe.drop(columns=existing_cols)

    def run(self):
        """
        Execute the full pipeline:
        1. Assign splits
        2. Filter groups based on thresholds
        3. Return cleaned train, validation, and test sets
        """
        self.assign_splits()
        self.filter_groups()
        train_df, valid_df, test_df = self.split_datasets()
        return (
            self.drop_columns(train_df),
            self.drop_columns(valid_df),
            self.drop_columns(test_df),
        )