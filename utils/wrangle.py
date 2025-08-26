import pandas as pd

def assign_season(date):
    month = date.month
    if month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Southwest Monsoon'
    elif month in [9, 10, 11]:
        return 'Post Monsoon'
    else:  # Dec, Jan
        return 'Winter'

def wrangle(df):
    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
    df[['Max_Price', 'Modal_Price']] = df[['Max_Price', 'Modal_Price']].astype(float)

    df['Is_VFPCK'] = df['Market'].str.contains('VFPCK', case=False)
    df['Season'] = df['Arrival_Date'].apply(assign_season)

    df = df.groupby(
        ['Commodity', 'Variety', 'Grade', 'Arrival_Date', 'Market', 'Season', 'Is_VFPCK'],
        as_index=False
        ).agg({
        'Modal_Price': 'mean',
        'Max_Price': 'mean',
        'Min_Price': 'mean'
        })
    
    df['Product_Type'] = df['Commodity'] + '|' + df['Variety'] + '|' + df['Grade']
    df['Variety_Type'] = df['Commodity'] + '|' + df['Variety']


    market_counts = df.groupby("Market")["Modal_Price"].count()
    #valid_markets = market_counts[market_counts > 500].index
    # df = df[df['Market'].isin(valid_markets)]
    
    df['Year'] = df['Arrival_Date'].dt.year
     # ensure 3 unique years per product/market
    # valid_years = df.groupby(['Product_Type', 'Market'])['Year'].transform('nunique') == 3
    # df = df[valid_years]

    product_type_counts = df['Product_Type'].value_counts()
    valid_product_types = product_type_counts[product_type_counts > 15].index
    df = df[df['Product_Type'].isin(valid_product_types)]

    df.drop(columns=['Variety', 'Grade'], inplace=True)     

    column_order = ['Product_Type', 'Commodity', 'Variety_Type', 'Arrival_Date', 'Market', 'Is_VFPCK', 'Season', 'Year', 'Modal_Price', 'Max_Price', 'Min_Price']
    df = df[column_order]

    df = df.sort_values(by=['Product_Type', 'Arrival_Date', 'Market']).reset_index(drop=True)
    return df
