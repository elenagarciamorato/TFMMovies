import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from matplotlib import pyplot as plt
from datetime import datetime

# Esquema tipo de datos

dtype_scheme ={'budget': np.int64, 'genres': np.object, 'homepage': np.str, 'id': np.int64, 'keywords': np.object,
               'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
               'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
               'revenue': np.int64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
               'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.int64}

# Change your working directory to where the data resides
os.chdir('/Users/elenagarciamorato/Desktop/TFM')

df = dd.read_csv('tmdb_5000_movies.csv', dtype=dtype_scheme)
#df = df.set_index('original_title')
print(df)

######## LIMPIEZA DE DATOS #########
# columns and number of rows
print("DataFrame columns: ")
print(df.columns)
print("DataFrame rows (number of): " + str(len(df)))

# NULL VALUES

# null values count
missing_values = df.isnull().sum()
# null values %
missing_count = ((missing_values / df.index.size) * 100)

# compute
with ProgressBar():
    missing_count_pct = missing_count.compute()
print(missing_count_pct)
#print(df.head())

# Drop not relevant columns and with high number of null values (+50%) -> homepage
#df = df.drop(columns=['tagline', 'homepage', 'id', 'overview', 'title', 'keywords'])
df = df.drop(columns=['homepage'])
print(df.columns)

# Fill missed values with a default one (between 5% and 50% null values)
#df = df.fillna({'release_date': 'Dismissed', 'runtime': 'Dismissed'})

# Discard rows with missing values (just in case null values were a very low number (<5%)) -> release_date and runtime
df = df.dropna(subset=('release_date', 'runtime'))


# EVALUATING VALUES

# values count
for i in (['budget', 'genres', 'id', 'keywords', 'original_language', 'original_title',
           'overview', 'popularity', 'production_companies', 'production_countries',
           'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline',
           'title', 'vote_average', 'vote_count']):
    values = df[i].value_counts().compute()
    print(values)

# Unique Values

# Drop columns with a high number of unique values -> 'tagline'(3944), 'id'(4800), 'overview'(4799), 'title'(4797)
# and 'keywords' - (4220) (despite their high level of unique values, we keep original-title as key value of each film)
df = df.drop(columns=['tagline', 'id', 'overview', 'title', 'keywords'])

# Drop rows whose status is not "Released" because only 3 of them are different,
# and then drop the status column because it doesn't make sense
df = df[(df.status == "Released")]
df = df.drop(columns=['status'])

# Odd Values

# Drop rows with budget == 0, runtime ==0
df = df[(df.budget != 0) & (df.runtime != 0)]



# OTHER

# Parse object into date -> Colums release_date
#print(df['release_date'].value_counts().compute())
release_date_parsed = df['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"), meta=datetime)
df = df.drop(columns=['release_date'])
df = df.assign(release_date=release_date_parsed)
#print(df['release_date'].head())

print("DataFrame (new) colums: ")  # 4800 (antes 4803)
print(df.columns)
print("DataFrame rows (new number of): " + str(len(df)))  # 14 (antes 20)

## Homepage(pagina web) -> muchos nulos
## tagline(slogan) (3944), id (4800), overview (sinopsis) (4799), title (4797) son casitodo valores unicos que no aportan info
## keywords -> no existe en el otro db y muchos valores unicos (4220) , ¿lo eliminamos?
##     status -> (released, rumored (5), post-production (3)) ¿Quitamos las q no estan released?y eliminamos esta columna?)
## ¿QUE HACEMOS CON POPULARITY??


# DESCRIPTIVE STATISTICS

print(df)


# Numeric value columns: budget, popularity, revenue, runtime, vote average, vote count

# Budget
print("\nBUDGET: ")
print(df['budget'].describe().compute())
#print("mean: " + str(df['budget'].mean().compute()))
#print("std: " + str(df['budget'].std().compute()))
#print("minimum: " + str(df['budget'].min().compute()))
#print("maximum: " + str(df['budget'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['budget'].values).compute())))

# Popularity
print("\nPOPULARITY: ")
print(df['popularity'].describe().compute())
#print("mean: " + str(df['popularity'].mean().compute()))
#print("std: " + str(df['popularity'].std().compute()))
#print("minimum: " + str(df['popularity'].min().compute()))
#print("maximum: " + str(df['popularity'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['popularity'].values).compute())))

# Revenue
print("\nREVENUE: ")
print(df['revenue'].describe().compute())
#print("mean: " + str(df['revenue'].mean().compute()))
#print("std: " + str(df['revenue'].std().compute()))
#print("minimum: " + str(df['revenue'].min().compute()))
#print("maximum: " + str(df['revenue'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['revenue'].values).compute())))

# Runtime
print("\nRUNTIME: ")
print(df['runtime'].describe().compute())
#print("mean: " + str(df['runtime'].mean().compute()))
#print("std: " + str(df['runtime'].std().compute()))
#print("minimum: " + str(df['runtime'].min().compute()))
#print("maximum: " + str(df['runtime'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['runtime'].values).compute())))

# Vote Average
print("\nVOTE AVERAGE: ")
print(df['vote_average'].describe().compute())
#print("mean: " + str(df['vote_average'].mean().compute()))
#print("std: " + str(df['vote_average'].std().compute()))
#print("minimum: " + str(df['vote_average'].min().compute()))
#print("maximum: " + str(df['vote_average'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['vote_average'].values).compute())))

# Vote Count
print("\nVOTE COUNT: ")
print(df['vote_count'].describe().compute())
#print("mean: " + str(df['vote_count'].mean().compute()))
#print("std: " + str(df['vote_count'].std().compute()))
#print("minimum: " + str(df['vote_count'].min().compute()))
#print("maximum: " + str(df['vote_count'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['vote_count'].values).compute())))


# String object columns: original_title


# Date type columns: release_date

# Columnas con objetos JSON: genres, original_language, spoken_languages, production_companies, production_countries
