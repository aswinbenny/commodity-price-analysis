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

def wrangle(df):

    df['Product_Type'] = df['Commodity'] + '|' + df['Variety'] + '|' + df['Grade']
    df['Product_Type'] = df['Product_Type'].str.replace('/', '_', regex=False)
    df['Variety_Type'] = df['Commodity'] + '|' + df['Variety']

    grp = df.groupby(['Product_Type', 'Market']).agg({'Arrival_Date': 'count'}).reset_index()
    products = grp[grp['Arrival_Date']>50][['Product_Type', 'Market']]
    df = df.merge(products, on=['Product_Type', 'Market'], how='inner')

    df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], format='%d/%m/%Y')
    df[['Max_Price', 'Modal_Price']] = df[['Max_Price', 'Modal_Price']].astype(float)


    df = df.groupby(
        ['Product_Type', 'Commodity', 'Variety_Type', 'Arrival_Date', 'Market'],
        as_index=False
        ).agg({
        'Modal_Price': 'mean',
        'Max_Price': 'mean',
        'Min_Price': 'mean'
        })
    
    df['Is_VFPCK'] = df['Market'].str.contains('VFPCK', case=False)
    df['Season'] = df['Arrival_Date'].apply(assign_season)
    df['Year'] = df['Arrival_Date'].dt.year
    df['log_Modal_Price'] = np.log1p(df['Modal_Price'])  

    column_order = ['Product_Type', 'Commodity', 'Variety_Type', 'Arrival_Date', 'Market', 'Is_VFPCK', 'Season', 'Year', 'Modal_Price', 'log_Modal_Price', 'Max_Price', 'Min_Price']
    df = df[column_order]

    df = df.sort_values(by=['Product_Type', 'Market', 'Arrival_Date']).reset_index(drop=True)
    return df
