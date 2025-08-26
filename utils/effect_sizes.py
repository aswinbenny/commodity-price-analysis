import numpy as np

def compute_effect_sizes_by_group(df, group_cols = ['Season', 'Market', 'Year']):
    for group_col in group_cols:
        for product, group in df.groupby('Product_Type'):   
            group['log_Modal_Price'] = group["Modal_Price"].apply(lambda X: np.log(X) if X > 0 else np.nan)
            prices = [grp['log_Modal_Price'].values for _, grp in group.groupby(group_col)]
            mean_prices = [np.mean(p) for p in prices]
            grand_mean = np.mean(np.concatenate(prices))
            size_prices = [len(p) for p in prices]
            variance_prices = [np.var(p, ddof=1) if len(p) > 1 else 0 for p in prices]
            SS_between = sum([(mean_prices[i] - grand_mean)**2 * size_prices[i] for i in range(len(prices))])
            SS_within = sum([(size_prices[i] - 1) * variance_prices[i] for i in range(len(prices))])
            SS_total = SS_between + SS_within
            eta2 = SS_between / SS_total if SS_total > 0 else 0
            df_between = len(prices) - 1
            df_within = sum(size_prices) - len(prices)
            MS_within = SS_within / df_within if df_within > 0 else 0
            omega2 = (SS_between - df_between * MS_within) / (SS_total + MS_within) if (SS_total + MS_within) > 0 else 0
            # Assign only to rows of this product
            df.loc[group.index, f'eta2_{group_col}'] = eta2
            df.loc[group.index, f'omega2_{group_col}'] = omega2
    return df


def compute_effect_sizes_by_commodity(df, group_cols=['Commodity', 'Variety_Type']):
    for group_col in group_cols:
        for commodity, group in df.groupby(group_col):
            group['Log_Modal_Price'] = group['Modal_Price'].apply(lambda x: np.log(x) if x > 0 else np.nan)
            prices = [grp['Log_Modal_Price'].values for _, grp in group.groupby('Product_Type')]
            mean_prices = [np.mean(p) for p in prices]
            grand_mean = np.mean(np.concatenate(prices))
            size_prices = [len(p) for p in prices]
            variance_prices = [np.var(p, ddof=1) if len(p) > 1 else 0 for p in prices]
            SS_between = sum([(mean_prices[i] - grand_mean)**2 * size_prices[i] for i in range(len(prices))])
            SS_within = sum([(size_prices[i] - 1) * variance_prices[i] for i in range(len(prices))])
            SS_total = SS_between + SS_within
            eta2 = SS_between / SS_total if SS_total > 0 else 0
            df_between = len(prices) - 1
            df_within = sum(size_prices) - len(prices)
            MS_within = SS_within / df_within if df_within > 0 else 0
            omega2 = (SS_between - df_between * MS_within) / (SS_total + MS_within) if (SS_total + MS_within) > 0 else 0
            df.loc[group.index, f'eta2_{group_col}'] = eta2
            df.loc[group.index, f'omega2_{group_col}'] = omega2
    return df
        

