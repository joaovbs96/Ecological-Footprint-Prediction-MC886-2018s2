# Import
# import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('NFA 2018.csv')

d = {}
d['country'] = []
d['ISO alpha-3 code'] = []
d['UN_region'] = []
d['UN_subregion'] = []
d['year'] = []
d['Percapita GDP (2010 USD)'] = []
d['population'] = []

records = ['BiocapTotGHA', 'EFConsTotGHA', 'EFExportsTotGHA', 'EFImportsTotGHA', 'EFProdTotGHA']
features = ['crop_land', 'grazing_land', 'forest_land', 'fishing_ground', 'built_up_land']

for r in records:
    for f in features:
        d[r + '_' + f] = []

d['carbon'] = []

# removendo combinações que não existem ou não fazem sentido
d.pop('EFExportsTotGHA_built_up_land')
d.pop('EFImportsTotGHA_built_up_land')

# create our new data frame
df = pd.DataFrame(data=d)
# add new features
# for each unique country
p = 0
for country in data.country.unique():
    # for each unique year
    for year in data.year.unique():
        # here we have all entries of a given year for a country
        yearly = data[(data.country == country) & (data.year == year)]

        if(len(yearly['country'].values) != 0):
            # create row to be added to our new dataset
            row = {}
            row['country'] = yearly['country'].values[0]
            row['ISO alpha-3 code'] = yearly['ISO alpha-3 code'].values[0]
            row['UN_region'] = yearly['UN_region'].values[0]
            row['UN_subregion'] = yearly['UN_subregion'].values[0]
            row['year'] = yearly['year'].values[0]
            row['Percapita GDP (2010 USD)'] = yearly['Percapita GDP (2010 USD)'].values[0]
            row['population'] = yearly['population'].values[0]

            carbon = 0.0
            for r in records:
                f_row = yearly[(yearly.record == r)]

                carbon += f_row['carbon'].values[0]

                for f in features:
                    key = r + '_' + f
                    if key in d:
                        row[r + '_' + f] = f_row[f].values[0]

            row['carbon'] = carbon

            df = df.append(row, ignore_index=True)
    p += 1
    print(str(p*100/(len(data.country.unique()))) + "%")

print(df.shape)
df.to_csv("NewNFA.csv", index=False)
