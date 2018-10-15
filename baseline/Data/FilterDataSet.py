# Import
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('NewNFA.csv')

# Drop unneeded features
data = data.drop(['ISO alpha-3 code'], 1)
data = data.drop(['UN_region'], 1)

# Drop World rows
data = data[data.UN_subregion != 'World']

# complete NAN values with linear interpolation
for country in data.country.unique():
    countryData = data[data.country == country]
    countryData = countryData.drop(['year'], 1)
    countryData = countryData.drop(['country'], 1)
    countryData = countryData.drop(['population'], 1)
    countryData = countryData.drop(['UN_subregion'], 1)

    countryData = countryData.interpolate(method='linear', limit_direction='both')
    data.update(countryData)

# remove remaining nans
data = data.dropna()

# shuffle data frame
data = data.sample(frac=1)

# save to file
data.to_csv("NewNFA-Filtered.csv", index=False)
