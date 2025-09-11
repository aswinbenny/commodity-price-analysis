import pandas as pd
import numpy as np

# Function to assign season based on month
def assign_season(date):
    """
    Maps a given date to a season.
    
    Seasons defined as:
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


def wrangle(df):
    """
    Performs initial wrangling of commodity price data.
    
    Steps included:
    1. Creates composite features for product identification:
       - Product_Type: Commodity|Variety|Grade
       - Variety_Type: Commodity|Variety
       Replaces '/' with '_' to avoid issues in downstream processing.
    2. Filters out (Product_Type, Market) pairs with less than 50 records
       to ensure meaningful trend analysis.
    3. Converts Arrival_Date to datetime and price columns to float.
    4. Aggregates data to daily mean prices per product-market combination.
    5. Adds features:
       - Is_VFPCK: boolean flag for VFPCK markets
       - Season: based on Arrival_Date
       - Year: extracted from Arrival_Date
       - log_Modal_Price: log-transformed Modal_Price for modeling stability
    6. Orders columns and sorts dataframe for consistency.
    
    Parameters:
        df (pd.DataFrame): Raw commodity price data with columns 
                           ['Commodity', 'Variety', 'Grade', 'Arrival_Date', 
                            'Market', 'Modal_Price', 'Max_Price', 'Min_Price']
    
    Returns:
        pd.DataFrame: Cleaned and feature-enhanced dataframe ready for EDA, 
                      statistical analysis, or modeling (Prophet/SARIMAX).
    """
    
    # Create composite features
    df['Product_Type'] = df['Commodity'] + '|' + df['Variety'] + '|' + df['Grade']
    df['Product_Type'] = df['Product_Type'].str.replace('/', '_', regex=False)
    df['Variety_Type'] = df['Commodity'] + '|' + df['Variety']

    # Filter product-market combinations with sufficient data
    grp = df.groupby(['Product_Type', 'Market']).agg({'Arrival_Date': 'count'}).reset_index()
    products = grp[grp['Arrival_Date'] > 50][['Product_Type', 'Market']]
    df = df.merge(products, on=['Product_Type', 'Market'], how='inner')

    # Convert columns to appropriate data types
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
    df[['Max_Price', 'Modal_Price']] = df[['Max_Price', 'Modal_Price']].astype(float)

    # Aggregate prices to daily mean per product-market
    df = df.groupby(
        ['Product_Type', 'Commodity', 'Variety_Type', 'Arrival_Date', 'Market'],
        as_index=False
    ).agg({
        'Modal_Price': 'mean',
        'Max_Price': 'mean',
        'Min_Price': 'mean'
    })

    # Add additional features
    df['Is_VFPCK'] = df['Market'].str.contains('VFPCK', case=False)
    df['Season'] = df['Arrival_Date'].apply(assign_season)
    df['Year'] = df['Arrival_Date'].dt.year
    df['log_Modal_Price'] = np.log1p(df['Modal_Price'])  # Log transform for modeling

    # Reorder columns for readability
    column_order = [
        'Product_Type', 'Commodity', 'Variety_Type', 'Arrival_Date', 'Market', 
        'Is_VFPCK', 'Season', 'Year', 'Modal_Price', 'log_Modal_Price', 
        'Max_Price', 'Min_Price'
    ]
    df = df[column_order]

    # Sort values for consistent indexing
    df = df.sort_values(by=['Product_Type', 'Market', 'Arrival_Date']).reset_index(drop=True)
    
    return df