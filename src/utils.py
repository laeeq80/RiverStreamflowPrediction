import pandas as pd
import numpy as np

df=pd.read_csv('data/intermidiate/cleaned.csv')

def date_to_date_month(df,Date):
    """
    Splits a datetime column into Day, Month, and Year.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the date column.
        date_column (str): Name of the date column.
    
    Returns:
        pd.DataFrame: Updated DataFrame with Day, Month, Year columns.
    """
    df[Date]=pd.to_datetime(df['Date'])
    df['Day']=df[Date].dt.day
    df['Month']=df[Date].dt.month
    df['year']=df[Date].dt.year
    
    """to convert the month and day to the sin and cos due to the cycle nature of the data to capture the seasonal behavior of the data such as january is near with december """
    
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df=df.drop(columns=['Month','Date','year','Day'])
    df.to_csv('data\\processed\\process.csv')
    return df
print(date_to_date_month(df,'Date'))

   