import pandas as pd
import matplotlib.pyplot as plt

def select_exemplars(df_grouped): 
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
        # High volume exemplar
        high_volumed_index = group['Total_Records'].idxmax()
        exemplar_row = group.loc[high_volumed_index]
        high_volume_exemplars[important_feature] = (
            exemplar_row['Product_Type'], 
            exemplar_row['Market']
        )

        # Representative exemplar
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
    rep_product, rep_market = representative_exemplars[feature]
    vol_product, vol_market = high_volume_exemplars[feature]

    df_rep = (df[(df['Product_Type'] == rep_product) & (df['Market'] == rep_market)]
              .set_index('Arrival_Date').sort_index())

    df_vol = (df[(df['Product_Type'] == vol_product) & (df['Market'] == vol_market)]
              .set_index('Arrival_Date').sort_index())

    return df_rep, df_vol


def plot_time_series(modal_price, title="Modal Price Trend"):
    plt.figure(figsize=(12, 6))
    plt.plot(modal_price, marker='o')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()