# Change your working directory to where the data resides
import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from datetime import datetime

# Esquema tipo de datos

dtype_scheme ={'budget': np.int64, 'genres': np.object, 'homepage': np.str, 'id': np.int64, 'keywords': np.object,
               'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
               'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
               'revenue': np.int64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
               'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.int64}

os.chdir('/Users/elenagarciamorato/Desktop/TFM')

df = dd.read_csv('tmdb_5000_movies.csv', dtype=dtype_scheme)
#df = df.set_index('original_title')
print(df)

######## LIMPIEZA DE DATOS #########
#columns and number of rows
print(df.columns)
print(len(df))

# NULL VALUES

# null values count
missing_values = df.isnull().sum()
#print(missing_values)

# null values %
missing_count = ((missing_values / df.index.size) * 100)
#print(missing_count)

#compute
with ProgressBar():
    missing_count_pct = missing_count.compute()
print(missing_count_pct)
#print(df.head())

## Tagline es el slogan y homepage la pagina web -> muchos nulos y no aportan ionformacion
## id, overview (sinopsis), title sontodo valores unicos que no aportan info
## status -> (released, rumored (5), post-production (3)) ¿Quitamos las q no estan released?y eliminamos esta columna?)
## keywords -> no existe en el otro db, ¿lo eliminamos?

# Drop not relevant columns and with high number of null values (+50%) -> homepage
#df = df.drop(columns=['tagline', 'homepage', 'id', 'overview', 'title', 'keywords'])
df = df.drop(columns=['homepage'])
print(df.columns)

# Fill missed values with a default one (between 5% and 50% null values)
#df = df.fillna({'release_date': 'Dismissed', 'runtime': 'Dismissed'})

# Discard rows with missing values (just in case null values were a very low number (<5%)) -> release_date and runtime
df = df.dropna(subset=('release_date', 'runtime'))


# UNIQUE VALUES

# unique values count
for i in ('tagline', 'id', 'overview', 'title', 'keywords'):
    unique_values=df[i].value_counts().compute()
    print(unique_values)

# Drop columns with a high number of unique values (+%) -> 'tagline', 'id', 'overview', 'title', 'keywords'
df = df.drop(columns=['tagline', 'id', 'overview', 'title', 'keywords'])

# OTHER

# Parse object into date -> Colums release_date
#print(df['release_date'].value_counts().compute())
release_date_parsed = df['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"), meta=datetime)
df = df.drop(columns=['release_date'])
df = df.assign(release_date=release_date_parsed)
#with ProgressBar(): print(df['release_date'].head())

# Descriptive statistics

# min max