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

######## DATA CLEANING #########
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

# Drop rows whose status is not "Released" because we don't care about not released films (8),
# and then drop the status column because it doesn't make sense
df = df[(df.status == "Released")]
df = df.drop(columns=['status'])


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

# Furthermore, on runtime we infer the value based on the median
median = (df['runtime'].quantile(0.5)).compute()
# print(" The median is: " + str(median))
df = df.fillna({'runtime': float(median)})

# Furthermore, on runtime we infer the value based on the neighbor method
# Imputed values only based in ['budget', 'popularity', 'revenue', 'runtime', 'vote average', 'vote count']
# df_aux = df.drop(columns= ['genres', 'spoken_languages', 'production_companies', 'production_countries', 'original_title', 'original_language', 'release_date'])
# imputer = KNNImputer(n_neighbors=3, weights="uniform")
# df_filled = imputer.fit_transform(df_aux)
# No es posible imputar valores con KNN, dado que el dtaframe a imputar no puede contener variables no numericas
# y ademas se devuelve un array



# OTHER

# Parse object into date -> release_date
#print(df['release_date'].value_counts().compute())
release_date_parsed = df['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"), meta=datetime)
df = df.drop(columns=['release_date'])
df = df.assign(release_date=release_date_parsed)
#print(df['release_date'].head())

# Reduce the number of different values (37) by putting unique entries (14) in a category called other -> original language
count_original_languages = df['original_language'].value_counts().compute()
single_language = list(count_original_languages[count_original_languages == 1].index)
condition = df['original_language'].isin(single_language)
original_language_masked = df['original_language'].mask(condition, 'other')
df = df.drop('original_language', axis=1)
df = df.assign(original_language=original_language_masked)

# Drop the column 'popularity' because doesn't propose relevant information
# (value calculated every different day by TMBD based on a unknown algorithm)
df = df.drop(columns=['popularity'])

# Convert str/Json columns to tuples -> genres, spoken_languages, production_companies, production_countries

# genres
genres_parsed = df['genres'].apply(lambda x: str_to_tuple(x, 'name'), meta=object)
df = df.drop(columns=['genres'])
df = df.assign(genres=genres_parsed)

# spoken_languages
spoken_languages_parsed = df['spoken_languages'].apply(lambda x: str_to_tuple(x, 'iso_639_1'), meta=object)
df = df.drop(columns=['spoken_languages'])
df = df.assign(spoken_languages=spoken_languages_parsed)

# production_companies
production_companies_parsed = df['production_companies'].apply(lambda x: str_to_tuple(x, 'name'), meta=object)
df = df.drop(columns=['production_companies'])
df = df.assign(production_companies=production_companies_parsed)

# production_countries
production_countries_parsed = df['production_countries'].apply(lambda x: str_to_tuple(x, 'name'), meta=object)
df = df.drop(columns=['production_countries'])
df = df.assign(production_countries=production_countries_parsed)

# Resume of cleaning data process
print("DataFrame (new) colums: ")  # 4792 (antes 4803)
print(df.columns)
print("DataFrame rows (new number of): " + str(len(df)))  # 13 (antes 20)

## Homepage(pagina web) -> muchos nulos
## tagline(slogan) (3944), id (4800), overview (sinopsis) (4799), title (4797) son casitodo valores unicos que no aportan info
## keywords -> no existe en el otro db y muchos valores unicos (4220)
## status -> (released, rumored (5), post-production (3)) Quitamos las q no estan released y eliminamos esta columna
## popularity -> popularidad de una pelicula en el momento de la extracción calculada por TMDB en base a un algoritmo no descrito



#### DESCRIPTIVE STATISTICS ####

#print(df)
# Numeric value columns: budget, popularity, revenue, runtime, vote average, vote count

# Budget
print("\nBUDGET: ")
print(df['budget'].describe().compute())
#print("mean: " + str(df['budget'].mean().compute()))
#print("std: " + str(df['budget'].std().compute()))
#print("minimum: " + str(df['budget'].min().compute()))
#print("maximum: " + str(df['budget'].max().compute()))
print("skewness: " + str(float(dask_stats.skew(df['budget'].values).compute())))

# # Popularity
# print("\nPOPULARITY: ")
# print(df['popularity'].describe().compute())
# #print("mean: " + str(df['popularity'].mean().compute()))
# #print("std: " + str(df['popularity'].std().compute()))
# #print("minimum: " + str(df['popularity'].min().compute()))
# #print("maximum: " + str(df['popularity'].max().compute()))
# print("skewness: " + str(float(dask_stats.skew(df['popularity'].values).compute())))

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

#df_test = df[(df.original_titlec== "Diamond Ruff")]
#print(df_test.runtime.head())

# values count
# for i in (['budget', 'genres', 'original_language', 'original_title',
#            'production_companies', 'production_countries',
#            'release_date', 'revenue', 'runtime', 'spoken_languages',
#            'vote_average', 'vote_count']):
#     values = df[i].value_counts().compute()
#     print(values)


# String object columns: original_title, original_language


# Date type columns: release_date

# JSON object columns: genres, spoken_languages, production_companies, production_countries


# SQL Querys

# Correlation between budget and revenue

with ProgressBar():
    bud_and_rev = df[['budget', 'revenue']]
    correlation_matrix = bud_and_rev.corr().compute()
print(correlation_matrix)

seaborn.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 10))
seaborn.despine(f, left=True, bottom=True)

# seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, robust=True)
seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, order=2)
plt.ylim(ymin=0)
plt.xlim(xmin=0)
#plt.ylim(ymax=1000000000)
#plt.xlim(xmax=1000000000)
plt.show()


