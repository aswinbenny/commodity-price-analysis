class DataSplitter:

    def __init__(self, df, date_dict: dict, thresholds: dict):
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
        """Add 'Split' column to df."""
        self.df['Split'] = self.df['Arrival_Date'].apply(lambda d: self.assign_split(d))
        
    def filter_groups(self):
        """Filter out groups with too few samples in any split."""
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
        """Return train/valid/test DataFrames."""
        train_df = self.df[self.df['Split'] == 'train']
        valid_df = self.df[self.df['Split'] == 'valid']
        test_df = self.df[self.df['Split'] == 'test']
        return train_df, valid_df, test_df
        
    
    def drop_columns(self, dataframe):
        """Drop leakage or helper columns safely."""
        columns_to_drop = ['Split', 'train', 'valid', 'test', 'Modal_Price', 'Max_Price', 'Min_Price']
        dataframe = dataframe.drop(columns=columns_to_drop)
        return dataframe

    def run(self):
        """Full pipeline: assign splits → filter groups → return cleaned splits."""
        self.assign_splits()
        self.filter_groups()
        train_df, valid_df, test_df = self.split_datasets()
        return (
            self.drop_columns(train_df),
            self.drop_columns(valid_df),
            self.drop_columns(test_df),
        )