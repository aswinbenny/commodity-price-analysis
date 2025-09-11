import numpy as np
import pandas as pd

def compute_effect_sizes_by_group(df, group_cols=['Season', 'Market', 'Year']):
    """
    Compute eta-squared (η²) and omega-squared (ω²) effect sizes for product-market combinations
    grouped by specified features (e.g., Season, Market, Year).
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Product_Type', 'Modal_Price', ...]
        group_cols (list): List of columns to compute effect sizes by.
    
    Returns:
        pd.DataFrame: Original DataFrame with added columns for eta2 and omega2 per group.
    """
    for group_col in group_cols:
        for product, group in df.groupby('Product_Type'):
            # Log-transform prices to stabilize variance
            group['log_Modal_Price'] = group["Modal_Price"].apply(lambda x: np.log(x) if x > 0 else np.nan)

            # Split the group by the current grouping column
            prices = [grp['log_Modal_Price'].values for _, grp in group.groupby(group_col)]
            mean_prices = [np.mean(p) for p in prices]
            grand_mean = np.mean(np.concatenate(prices))
            size_prices = [len(p) for p in prices]
            variance_prices = [np.var(p, ddof=1) if len(p) > 1 else 0 for p in prices]

            # Compute sum of squares
            SS_between = sum([(mean_prices[i] - grand_mean)**2 * size_prices[i] for i in range(len(prices))])
            SS_within = sum([(size_prices[i] - 1) * variance_prices[i] for i in range(len(prices))])
            SS_total = SS_between + SS_within

            # Effect size calculations
            eta2 = SS_between / SS_total if SS_total > 0 else 0
            df_between = len(prices) - 1
            df_within = sum(size_prices) - len(prices)
            MS_within = SS_within / df_within if df_within > 0 else 0
            omega2 = (SS_between - df_between * MS_within) / (SS_total + MS_within) if (SS_total + MS_within) > 0 else 0

            # Assign effect sizes to original DataFrame rows
            df.loc[group.index, f'eta2_{group_col}'] = eta2
            df.loc[group.index, f'omega2_{group_col}'] = omega2

    return df


def compute_effect_sizes_by_commodity(df, group_cols=['Commodity', 'Variety_Type']):
    """
    Compute eta-squared (η²) and omega-squared (ω²) effect sizes for product groups
    within each commodity or variety type.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Product_Type', 'Modal_Price', ...]
        group_cols (list): List of columns (Commodity or Variety_Type) to group by.
    
    Returns:
        pd.DataFrame: Original DataFrame with added eta2 and omega2 per commodity/variety.
    """
    for group_col in group_cols:
        for _, group in df.groupby(group_col):
            # Log-transform prices
            group['Log_Modal_Price'] = group['Modal_Price'].apply(lambda x: np.log(x) if x > 0 else np.nan)

            # Split by product type within the group
            prices = [grp['Log_Modal_Price'].values for _, grp in group.groupby('Product_Type')]
            mean_prices = [np.mean(p) for p in prices]
            grand_mean = np.mean(np.concatenate(prices))
            size_prices = [len(p) for p in prices]
            variance_prices = [np.var(p, ddof=1) if len(p) > 1 else 0 for p in prices]

            # Compute sum of squares
            SS_between = sum([(mean_prices[i] - grand_mean)**2 * size_prices[i] for i in range(len(prices))])
            SS_within = sum([(size_prices[i] - 1) * variance_prices[i] for i in range(len(prices))])
            SS_total = SS_between + SS_within

            # Effect sizes
            eta2 = SS_between / SS_total if SS_total > 0 else 0
            df_between = len(prices) - 1
            df_within = sum(size_prices) - len(prices)
            MS_within = SS_within / df_within if df_within > 0 else 0
            omega2 = (SS_between - df_between * MS_within) / (SS_total + MS_within) if (SS_total + MS_within) > 0 else 0

            df.loc[group.index, f'eta2_{group_col}'] = eta2
            df.loc[group.index, f'omega2_{group_col}'] = omega2

    return df


def compare_eta_omega(df, features=['Commodity', 'Variety_Type', 'Season', 'Market', 'Year'],
                      absolute_differnece_threshold=0.1, ratio_threshold=1.5):
    """
    Compare eta2 and omega2 effect sizes to identify suspiciously large differences
    which may indicate unstable effect size estimation.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing eta2 and omega2 columns per feature.
        features (list): Features to compare effect sizes on.
        absolute_differnece_threshold (float): Minimum difference to flag.
        ratio_threshold (float): Minimum ratio to flag.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame of flagged feature groups.
    """
    results = []
    for feature in features:
        df[f'eta2-omega2_diff_{feature}'] = df[f'eta2_{feature}'] - df[f'omega2_{feature}']
        df[f'eta2/omega2_ratio_{feature}'] = df[f'eta2_{feature}'] / (df[f'omega2_{feature}'] + 1e-9)
        
        suspicious = df[
            (df[f'eta2-omega2_diff_{feature}'] > absolute_differnece_threshold) &
            (df[f'eta2/omega2_ratio_{feature}'] > ratio_threshold)
        ].copy()
        suspicious['feature_flagged'] = feature
        results.append(suspicious)

    if results:
        suspicious_df = pd.concat(results)
    else:
        suspicious_df = pd.DataFrame()

    return suspicious_df