# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2019/2020

import os

from functions import *
import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn
from sklearn.impute import KNNImputer


# Datatype scheme
dtype_scheme ={'budget': np.float64, 'genres': np.object, 'homepage': np.str, 'id': np.float64, 'keywords': np.object,
               'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
               'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
               'revenue': np.float64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
               'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.float64}

# Change your working directory to where the data resides
os.chdir('/Users/elenagarciamorato/Desktop/TFM')

df = dd.read_csv('tmdb_5000_movies.csv', dtype=dtype_scheme)
#df = df.set_index('original_title')
print(df)

######## DATA CLEANING #########

# columns and number of rows
print("DataFrame columns: " + str(df.columns))
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
columns_to_drop = list(missing_count_pct[missing_count_pct >= 50].index)
df = df.drop(columns_to_drop, axis=1)

# Fill missed values with a default one (between 5% and 50% null values)
#df = df.fillna({'release_date': 'Dismissed', 'runtime': 'Dismissed'})

# Discard rows with missing values (just in case null values were a very low number (<5%)) -> release_date, runtime, overview
rows_to_drop = list(missing_count_pct[(missing_count_pct > 0) & (missing_count_pct < 5)].index)
df = df.dropna(subset=rows_to_drop)


# EVALUATING VALUES

# values count
for i in (['budget', 'genres', 'id', 'keywords', 'original_language', 'original_title',
           'overview', 'popularity', 'production_companies', 'production_countries',
           'release_date', 'revenue', 'runtime', 'spoken_languages', 'status', 'tagline',
           'title', 'vote_average', 'vote_count']):
    values = df[i].value_counts().compute()
    print(values)

# Unique Values

# Drop columns with a high number of unique values -> 'tagline'(3944), 'overview'(4799), 'title'(4797)
# and 'keywords' - (4220) (despite their high level of unique values, we keep id as key value of each film
# and original-title to identify them in inequivocous way)
df = df.drop(columns=['tagline', 'overview', 'title', 'keywords'])


# Odd Values


# Value 0 in columns 'budget', 'revenue' and 'runtime' doesn't mean anything but missed values
# To fix it, on budget, revenue and runtime (int64) we set the 0 values as NaN
condition = df['budget'] == 0
budget_masked = df['budget'].mask(condition, np.nan)
df = df.drop('budget', axis=1)
df = df.assign(budget=budget_masked)

condition = df['revenue'] == 0
revenue_masked = df['revenue'].mask(condition, np.nan)
df = df.drop('revenue', axis=1)
df = df.assign(revenue=revenue_masked)

condition = df['runtime'] == 0
runtime_masked = df['runtime'].mask(condition, np.nan)
df = df.drop('runtime', axis=1)
df = df.assign(runtime=runtime_masked)

# Again, we evaluate null values
with ProgressBar():
    missing_count_pct = ((df.isnull().sum() / df.index.size) * 100).compute()
print(missing_count_pct)

# Furthermore, on runtime we infer the value based on the median
median = (df['runtime'].quantile(0.5)).compute()
print(" The median is: " + str(median))
df = df.fillna({'runtime': float(median)})

# Furthermore, on runtime we infer the value based on the neighbor method
# Imputed values only based in ['budget', 'popularity', 'revenue', 'runtime', 'vote average', 'vote count']
#df_aux = df.drop(columns=['genres', 'spoken_languages', 'production_companies', 'production_countries', 'original_title', 'original_language', 'release_date'])
#imputer = KNNImputer(n_neighbors=5, weights="uniform")
#df_filled = imputer.fit_transform(df_aux)
# No es posible imputar valores con KNN, dado que el dtaframe a imputar no puede contener variables no numericas
# y ademas se devuelve un array



# OTHER

# Parse object into date and keep only the year-> release_date
release_year = df['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").year, meta=datetime)
df = df.drop(columns=['release_date'])
df = df.assign(release_year=release_year)
print(df['release_year'].head())

# Reduce the number of different values (37) by putting unique entries (14) in a category called other -> original language
count_original_languages = df['original_language'].value_counts().compute()
single_language = list(count_original_languages[count_original_languages == 1].index)

condition = df['original_language'].isin(single_language)
original_language_masked = df['original_language'].mask(condition, 'other')
df = df.drop('original_language', axis=1)
df = df.assign(original_language=original_language_masked)

# Drop rows whose status is not "Released" because we don't care about not released films (8),
# and then drop the status column because it doesn't make sense
df = df[(df.status == "Released")]
df = df.drop(columns=['status'])

# Drop the column 'popularity' because doesn't propose relevant information
# (value calculated every different day by TMBD based on a unknown algorithm)
df = df.drop(columns=['popularity'])


# Convert str/Json columns to tuples -> genres, spoken_languages, production_companies, production_countries

# genres
genres_parsed = df['genres'].apply(lambda x: get_list(x, 'name'), meta=object)
df = df.drop(columns=['genres'])
df = df.assign(genres=genres_parsed)

# spoken_languages
spoken_languages_parsed = df['spoken_languages'].apply(lambda x: get_list(x, 'iso_639_1'), meta=object)
df = df.drop(columns=['spoken_languages'])
df = df.assign(spoken_languages=spoken_languages_parsed)

# production_companies
production_companies_parsed = df['production_companies'].apply(lambda x: get_list(x, 'name'), meta=object)
df = df.drop(columns=['production_companies'])
df = df.assign(production_companies=production_companies_parsed)

# production_countries
production_countries_parsed = df['production_countries'].apply(lambda x: get_list(x, 'name'), meta=object)
df = df.drop(columns=['production_countries'])
df = df.assign(production_countries=production_countries_parsed)

# Resume of cleaning data process
print("DataFrame (new) colums: ")   # 13 (antes 20)
print(df.columns)
print("DataFrame rows (new number of): " + str(len(df)))  # 4792 (antes 4803)

## Homepage(pagina web) -> muchos nulos
## tagline(slogan) (3944), id (4800), overview (sinopsis) (4799), title (4797) son casitodo valores unicos que no aportan info
## keywords -> no existe en el otro db y muchos valores unicos (4220)
## status -> (released, rumored (5), post-production (3)) Quitamos las q no estan released y eliminamos esta columna
## popularity -> popularidad de una pelicula en el momento de la extracción calculada por TMDB en base a un algoritmo no descrito



#### DESCRIPTIVE STATISTICS ####

#print(df)
# Numeric value columns: budget, revenue, runtime, vote average, vote count
"""
# Budget
print("\nBUDGET: ")
print(df['budget'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df['budget'].values).compute())))


# Revenue
print("\nREVENUE: ")
print(df['revenue'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df['revenue'].values).compute())))

# Runtime
print("\nRUNTIME: ")
print(df['runtime'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df['runtime'].values).compute())))

# Vote Average
print("\nVOTE AVERAGE: ")
print(df['vote_average'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df['vote_average'].values).compute())))

# Vote Count
print("\nVOTE COUNT: ")
print(df['vote_count'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df['vote_count'].values).compute())))


# String object columns: original_title, original_language
print("\nORIGINAL LANGUAGE: ")
print(df['original_language'].describe().compute())

print("\nORIGINAL TITLE: ")
print(df['original_title'].describe().compute())

# Date type columns: release_date
print("\nRELEASE YEAR: ")
print(df['release_year'].describe().compute())

"""

# SQL QUERYS


