import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def select_exemplars(df_grouped): 
    """
    Select representative and high-volume exemplars for each important feature group.
    
    Representative exemplars are closest to the mean effect sizes.
    High-volume exemplars are those with the highest total records.
    
    Parameters:
        df_grouped (pd.DataFrame): Grouped DataFrame containing 'important_features',
                                   'Product_Type', 'Market', 'Total_Records', and effect size columns.
    
    Returns:
        tuple: (representative_exemplars, high_volume_exemplars)
            Each is a dictionary mapping feature -> (Product_Type, Market)
    """
    representative_exemplars = {}
    high_volume_exemplars = {}

    effect_size_columns = [
        'Mean_Commodity_Effect_Size', 
        'Mean_Variety_Type_Effect_Size', 
        'Mean_Season_Effect_Size', 
        'Mean_Year_Effect_Size', 
        'Mean_Market_Effect_Size'
    ]
    
    for important_feature, group in df_grouped.groupby('important_features'):
        # High-volume exemplar: product-market with most records
        high_volumed_index = group['Total_Records'].idxmax()
        exemplar_row = group.loc[high_volumed_index]
        high_volume_exemplars[important_feature] = (
            exemplar_row['Product_Type'], 
            exemplar_row['Market']
        )

        # Representative exemplar: product-market closest to mean effect sizes
        group_means = group[effect_size_columns].mean()
        distances = group[effect_size_columns].sub(group_means).abs().sum(axis=1)
        representative_index = distances.idxmin()
        exemplar_row = group.loc[representative_index]
        representative_exemplars[important_feature] = (
            exemplar_row['Product_Type'], 
            exemplar_row['Market']
        )
    
    return representative_exemplars, high_volume_exemplars


def time_series_extractor(df, feature, representative_exemplars, high_volume_exemplars):
    """
    Extract time series data for the representative and high-volume exemplars of a feature.
    
    Parameters:
        df (pd.DataFrame): Full dataset with 'Product_Type', 'Market', and 'Arrival_Date'.
        feature (str): Feature name to extract exemplars for.
        representative_exemplars (dict): Output of select_exemplars.
        high_volume_exemplars (dict): Output of select_exemplars.
    
    Returns:
        tuple: (df_rep, df_vol)
            DataFrames indexed by 'Arrival_Date' for representative and high-volume exemplars.
    """
    rep_product, rep_market = representative_exemplars[feature]
    vol_product, vol_market = high_volume_exemplars[feature]

    # Representative exemplar time series
    df_rep = (df[(df['Product_Type'] == rep_product) & (df['Market'] == rep_market)]
              .set_index('Arrival_Date').sort_index())

    # High-volume exemplar time series
    df_vol = (df[(df['Product_Type'] == vol_product) & (df['Market'] == vol_market)]
              .set_index('Arrival_Date').sort_index())

    return df_rep, df_vol


def plot_time_series(modal_price, title="Modal Price Trend"):
    """
    Plot a time series of modal prices.
    
    Parameters:
        modal_price (pd.Series or pd.DataFrame column): Time-indexed prices to plot.
        title (str): Plot title.
    """
    plt.figure(figsize=(24,10))
    plt.plot(modal_price, marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())            # Tick every month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y')) # Format: "Jan 2025"
    plt.grid(True)
    plt.show()