import pandas as pd
import numpy as np

# Function to assign season based on month
def assign_season(date):
    """
    Maps a given date to a season.

    Seasons:
    - March to May      -> Summer
    - June to August    -> Southwest Monsoon
    - September to Nov  -> Post Monsoon
    - December to Feb   -> Winter
    """
    month = date.month
    if month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Southwest Monsoon'
    elif month in [9, 10, 11]:
        return 'Post Monsoon'
    else:
        return 'Winter'


def wrangle_ml(df):
    """
    Wrangles commodity price data for machine learning models (e.g., LightGBM).

    Key steps:
    1. Creates composite features for product identification:
       - Product_Type = Commodity|Variety|Grade
    2. Converts Arrival_Date to datetime and Modal_Price to float.
    3. Drops unnecessary columns to focus on modeling-relevant data.
    4. Aggregates data per day for each product-market combination.
    5. Log-transforms Modal_Price to stabilize variance.
    6. Fills missing dates and prices for time series consistency:
       - Forward-fill up to 3 days, then interpolate.
    7. Generates non-leaky features for modeling:
       - Lag features (1,3,7,14,30 days)
       - Rolling mean & std (3,7,14 days) with shift(1) to prevent leakage
       - Expanding mean & std per product-market with shift(1)
    8. Extracts additional temporal and categorical features:
       - Commodity, Variety_Type, Is_VFPCK, Season, Year
       - Market_Season, month, day_of_week, day_of_month, quarter
       - Product_Month
       - is_weekend flag
    9. Sorts the dataframe for consistency.

    Parameters:
        df (pd.DataFrame): Raw commodity price data with relevant columns.

    Returns:
        pd.DataFrame: Feature-enhanced dataframe ready for ML modeling.
    """

    # Composite product features
    df['Product_Type'] = df['Commodity'] + '|' + df['Variety'] + '|' + df['Grade']
    df['Product_Type'] = df['Product_Type'].str.replace('/', '_', regex=False)

    # Convert data types
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
    df['Modal_Price'] = df['Modal_Price'].astype(float)

    # Drop columns not required for ML
    df = df.drop(columns=['State', 'District', 'Commodity', 'Variety', 'Grade', 
                          'Min_Price', 'Max_Price', 'Commodity_Code'])

    # Aggregate prices per day-product-market
    df = df.groupby(['Product_Type', 'Arrival_Date', 'Market'], as_index=False).agg({
        'Modal_Price': 'mean'
    })
    # Keep log_Modal_Price intact to preserve original logged price values before filling missing data
    df['log_Modal_Price'] = np.log1p(df['Modal_Price'])

    # Initialize list to store processed groups
    dfs = []
    for (prod, market), group in df.groupby(['Product_Type', 'Market']):
        # Reindex to ensure all dates are present
        group = group.set_index('Arrival_Date').reindex(
            pd.date_range(group['Arrival_Date'].min(), group['Arrival_Date'].max())
        )
        group['Product_Type'] = prod
        group['Market'] = market

        # Fill missing prices (forward fill + interpolate)
        group['Modal_Price_filled'] = group['Modal_Price'].ffill(limit=3).interpolate()
        group['log_Modal_Price_filled'] = np.log1p(group['Modal_Price_filled'])

        # Lag features (non-leaky)
        for lag in [1, 3, 7, 14, 30]:
            group[f'lag_{lag}'] = group['log_Modal_Price_filled'].shift(lag)

        # Rolling features with shift to prevent leakage
        for w in [3, 7, 14]:
            group[f'rolling_mean_{w}'] = group['log_Modal_Price_filled'].shift(1).rolling(w, min_periods=1).mean()
            group[f'rolling_std_{w}'] = group['log_Modal_Price_filled'].shift(1).rolling(w, min_periods=1).std()

        dfs.append(group)

    # Combine all product-market groups
    df = pd.concat(dfs).reset_index().rename(columns={'index': 'Arrival_Date'})

    # Additional categorical and temporal features
    df['Commodity'] = df['Product_Type'].apply(lambda x: x.split('|')[0])
    df['Variety_Type'] = df['Product_Type'].apply(lambda x: '|'.join(x.split('|')[:2]))
    df['Is_VFPCK'] = df['Market'].str.contains('VFPCK', case=False)
    df['Season'] = df['Arrival_Date'].apply(assign_season)
    df['Year'] = df['Arrival_Date'].dt.year
    df['Market_Season'] = df['Market'] + '|' + df['Season']
    df['month'] = df['Arrival_Date'].dt.month
    df['day_of_week'] = df['Arrival_Date'].dt.dayofweek  # Monday=0
    df['day_of_month'] = df['Arrival_Date'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['Arrival_Date'].dt.quarter
    df['Product_Month'] = df['Product_Type'] + '_' + df['month'].astype(str)

    # Sort and compute expanding features (non-leaky)
    df = df.sort_values(by=['Product_Type', 'Market', 'Arrival_Date']).reset_index(drop=True)
    df['exp_mean_pm'] = df.groupby(['Product_Type', 'Market'])['log_Modal_Price_filled']\
                            .transform(lambda x: x.expanding().mean().shift(1))
    df['exp_std_pm'] = df.groupby(['Product_Type', 'Market'])['log_Modal_Price_filled']\
                           .transform(lambda x: x.expanding().std().shift(1))

    return df