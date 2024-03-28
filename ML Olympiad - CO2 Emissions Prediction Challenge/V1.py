# Import libraries
import pandas as pd
import numpy as np
import random
import os
from tqdm.notebook import tqdm

# Geospatial libraries
import geopandas as gpd
from shapely.geometry import Point
import folium

from statsmodels.tsa.arima.model import ARIMA
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries (potential)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Pandas display options
pd.options.display.float_format = '{:.5f}'.format
pd.options.display.max_rows = None

# Jupyter notebook environment (potential)
%matplotlib inline

# Warnings (not recommended for production)
import warnings
warnings.filterwarnings('ignore')


# Set seed for reproducability

SEED = 2023
random.seed(SEED)
np.random.seed(SEED)

Train=pd.read_csv('/kaggle/input/ml-olympiad-co2-emissions-prediction-challenge/train.csv')
Test= pd.read_csv('/kaggle/input/ml-olympiad-co2-emissions-prediction-challenge/test.csv')
sample_submission= pd.read_csv('/kaggle/input/ml-olympiad-co2-emissions-prediction-challenge/sample_submission.csv')



#Filter BY CO2 emissions (metric tons per capita)
specific_string = 'CO2 emissions (metric tons per capita)'
filtered_rows = Train[Train['Indicator'] == specific_string]
filtered_rows

#Drop Indicator Column
filtered_rows.drop(columns=['Indicator'], inplace=True)
filtered_rows

#Drop Country Code Column
filtered_rows.drop(columns=['Country Code'], inplace=True)
filtered_rows

#Covert Year Column in rows
melted_df = filtered_rows.melt(id_vars='Country Name', var_name='Year', value_name='CO2_Emissions')
melted_df


#Sort Row in accending order by Year and Country Name
sorted_df = melted_df.groupby('Country Name').apply(lambda x: x.sort_values('Year')).reset_index(drop=True)
sorted_df.head(20)

#All country name into unique_values
unique_values = melted_df['Country Name'].unique()
unique_values.shape

#Drop all th row that have value ".."
sorted_df = sorted_df[sorted_df.ne('..').all(axis=1)]


sorted_df.head(15)

#forcasting Using ARIMA
for i in unique_values:
    afg = sorted_df[sorted_df['Country Name'] == i]
    
    
    print(i)
    country_data=afg['CO2_Emissions'].astype(float)
    try:
        model = ARIMA(country_data, order=(1,1,0))  # Adjust order as needed
        model_fit = model.fit()

    except:
        l=0
    forecast = model_fit.forecast(steps=16)
    row_index = sample_submission[sample_submission.eq(i).any(axis=1)].index[0]


    new_values = [i, forecast[16], forecast[17],forecast[18], forecast[19], forecast[20],forecast[30]]
    sample_submission.loc[row_index] = new_values
    

sample_submission

sample_submission.to_csv('submission.csv', index=False)
