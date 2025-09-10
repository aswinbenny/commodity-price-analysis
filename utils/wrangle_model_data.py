import pandas as pd
import numpy as np

def assign_season(date):
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
   
    df['Product_Type'] = df['Commodity'] + '|' + df['Variety'] + '|' + df['Grade']
    df['Product_Type'] = df['Product_Type'].str.replace('/', '_', regex=False)

    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
    df['Modal_Price'] = df['Modal_Price'].astype(float)
    df['log_Modal_Price'] = np.log1p(df['Modal_Price'])
 
    df = df.drop(columns =['State', 'District', 'Commodity', 'Variety',	'Grade', 'Min_Price', 'Max_Price', 'Commodity_Code'])

    df = df.groupby(
        ['Product_Type', 'Arrival_Date', 'Market'],
        as_index=False
        ).agg({
        'Modal_Price': 'mean',
        })

    dfs = []
    for (prod, market), group in df.groupby(['Product_Type', 'Market']):
        group = group.set_index('Arrival_Date').reindex(
            pd.date_range(group['Arrival_Date'].min(), group['Arrival_Date'].max())
        )
        group['Product_Type'] = prod
        group['Market'] = market
        group['log_Modal_Price_filled'] = np.log1p(group['Modal_Price']).ffill(limit=3).interpolate()

        for lag in [1,3,7,14,30]:
            group[f'lag_{lag}'] = group['log_Modal_Price_filled'].shift(lag)
        
        for w in [3,7,14]:
            group[f'rolling_mean_{w}'] = group['log_Modal_Price_filled'].rolling(w, min_periods=1).mean()
            group[f'rolling_std_{w}'] = group['log_Modal_Price_filled'].rolling(w, min_periods=1).std()

        dfs.append(group)

    df = pd.concat(dfs).reset_index().rename(columns={'index': 'Arrival_Date'})


    
    df['Commodity'] = df['Product_Type'].apply(lambda x: x.split('|')[0])
    df['Variety_Type'] = df['Product_Type'].apply(lambda x: '|'.join(x.split('|')[:2]))
    df['Is_VFPCK'] = df['Market'].str.contains('VFPCK', case=False)
    df['Season'] = df['Arrival_Date'].apply(assign_season)
    df['Year'] = df['Arrival_Date'].dt.year

    df = df.sort_values(by=['Product_Type', 'Market', 'Arrival_Date']).reset_index(drop=True)
    return df
