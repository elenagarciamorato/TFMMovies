# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2019/2020

import os

from functions import *
from querys import *
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from matplotlib import pyplot as plt
from datetime import datetime
import seaborn
from sklearn.impute import KNNImputer


# Datatype scheme
dtype_scheme = {'budget': np.object, 'genres': np.str, 'homepage': np.str, 'id': np.object,
                'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.object,
                 'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
                 'revenue': np.float64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
                 'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.float64,
                 'adult': np.object, 'belongs_to_collection': np.object, 'imdb_id': np.object, 'poster_path': np.object,
                'video': np.object}


# dtype_scheme3 = {'budget': np.int64, 'genres': np.object, 'homepage': np.str, 'id': np.int64,
#                  'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
#                 'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
#                  'revenue': np.int64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
#                  'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.int64,
#                 'adult': bool, 'belongs_to_collection': np.object, 'imdb_id': np.object, 'poster_path': np.object,
#                  'video': bool}

# Change your working directory to where the data resides
os.chdir('/Users/elenagarciamorato/Desktop/TFM')

df2 = dd.read_csv('movies_metadata.csv', dtype=dtype_scheme, error_bad_lines=False, sep=',',
                  encoding='unicode_escape')
#na_values=[' ', "NA", '[]', np.nan]
#df = df.set_index('id')
print(df2)

######## DATA CLEANING #########
# columns and number of rows
print("DataFrame columns: ")
print(df2.columns)
print("DataFrame rows (number of): " + str(len(df2.index)))


## NULL VALUES

# null values count
missing_values = df2.isnull().sum()
# null values %
missing_count = ((missing_values / df2.index.size) * 100)

# Counting missing values
with ProgressBar():
    missing_count_pct = missing_count.compute()
print(missing_count_pct)


# Drop columns with high number of null values (+50%) except 'belongs_to_collection'
# because null values means "doesn't belongs to any collection" -> homepage, tagline
columns_to_drop = list(missing_count_pct[missing_count_pct >= 50].index)
columns_to_drop.remove('belongs_to_collection')
df2 = df2.drop(columns_to_drop, axis=1)

# Discard rows with missing values (just in case null values were a very low number (<5%)) ->
# 'imdb_id', 'original_language', 'overview', 'popularity', 'poster_path', 'release_date', 'revenue',
# 'runtime', 'spoken_languages', 'status', 'title', 'video', 'vote_average', 'vote_count'))
rows_to_drop = list(missing_count_pct[(missing_count_pct > 0) & (missing_count_pct < 5)].index)
df2 = df2.dropna(subset=rows_to_drop)


# EVALUATING COLUMN VALUES

# values count
for i in df2.columns:
    values = df2[i].value_counts().compute()
    print(values)


# Unique Values

# Drop columns with a high number of unique values -> 'id'(44014), 'imdb_id'(44014), 'original_title' (42000),
# 'overview'(43851), 'poster_path'(43989), 'title'(40938)
# (despite their high level of unique values, we keep id as key value of each film
# and original-title to identify them in inequivocous way)
columns_unique_values = ['imdb_id', 'overview', 'poster_path', 'title']
df2 = df2.drop(columns=columns_unique_values)


# Other Values

# Drop rows whose key value (id) is duplicated (Just in case)
df2 = df2.drop_duplicates(['id'], keep='first')

# Set np.object column as numeric (float) -> budget
budget_parsed = df2['budget'].apply(lambda x: to_float(x), meta=np.float)
df2 = df2.drop(columns=['budget'])
df2 = df2.assign(budget=budget_parsed)

# Parse object into date and keep only the year-> release_date
release_year = df2['release_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").year, meta=datetime)
df2 = df2.drop(columns=['release_date'])
df2 = df2.assign(release_year=release_year)
print(df2['release_year'].head())


# Reduce the number of different values (89) by putting unique entries (16) in a category called other -> original language
count_original_languages = df2['original_language'].value_counts().compute()
single_language = list(count_original_languages[count_original_languages == 1].index)
condition = df2['original_language'].isin(single_language)
original_language_masked = df2['original_language'].mask(condition, 'other')
df2 = df2.drop('original_language', axis=1)
df2 = df2.assign(original_language=original_language_masked)

# Drop rows whose status is not "Released" because we don't care about not released films (338),
# and then drop the status column because it doesn't make sense
df2 = df2[(df2.status == "Released")]
df2 = df2.drop(columns=['status'])

# Drop the column 'popularity' because doesn't propose relevant information
# (value calculated every different day by TMBD based on a unknown algorithm)
df2 = df2.drop(columns=['popularity'])

# Drop the column 'video' because doesn't propose relevant information, only shows if there is a video
# linked to a film entry
df2 = df2.drop(columns=['video'])


# Odd Values

# Value 0 in columns 'budget', 'revenue' and 'runtime' doesn't mean anything but missed values
# To fix it, on budget, revenue and runtime (int64) we set the 0 values as NaN
condition = df2['budget'] == 0
budget_masked = df2['budget'].mask(condition, np.nan)
df2 = df2.drop('budget', axis=1)
df2 = df2.assign(budget=budget_masked)

condition = df2['revenue'] == 0
revenue_masked = df2['revenue'].mask(condition, np.nan)
df2 = df2.drop('revenue', axis=1)
df2 = df2.assign(revenue=revenue_masked)

condition = df2['runtime'] == 0
runtime_masked = df2['runtime'].mask(condition, np.nan)
df2 = df2.drop('runtime', axis=1)
df2 = df2.assign(runtime=runtime_masked)

# Again, we evaluate null values
with ProgressBar():
    missing_count_pct = ((df2.isnull().sum() / df2.index.size) * 100).compute()
print(missing_count_pct)

# And just as before, we drop columns with high number of missed values
#columns_to_drop = list(missing_count_pct[missing_count_pct >= 50].index)
#columns_to_drop.remove('belongs_to_collection')
#df2 = df2.drop(columns_to_drop, axis=1)

# Furthermore, on Runtime column (2,32% missed values) we infer the value based on the neighbor method (KNN)
# Imputed values only based in numeric entries ['budget', id, 'revenue', 'runtime', 'vote average', 'vote count', 'release_year']
df2_aux = df2.drop(columns=['genres', 'spoken_languages', 'production_companies', 'production_countries',
                            'original_title', 'original_language', 'belongs_to_collection', 'adult'])

imputer = KNNImputer(n_neighbors=5, weights="uniform", copy=True)
df2_aux = pd.DataFrame(imputer.fit_transform(df2_aux), columns=df2_aux.columns, index=df2_aux['id'])
# df2_aux = dd.from_pandas(df2_aux, npartitions=df2.npartitions)

df2 = df2.set_index('id')
df2 = df2.assign(runtime=df2_aux['runtime'])
df2 = df2.reset_index()

# Furthermore, on runtime we infer the value based on the median
#median = (df2['runtime'].quantile(0.5)).compute()
#print(" The median is: " + str(median))
#df2 = df2.fillna({'runtime': float(median)})



# CONVERT STR/JSON OBJECTS TO TUPLES-> genres, spoken_languages, production_companies, production_countries,
# belongs_to_collection

# genres
genres_parsed = df2['genres'].apply(lambda x: get_list(x, 'name'), meta=object)
df2 = df2.drop(columns=['genres'])
df2 = df2.assign(genres=genres_parsed)

# spoken_languages
spoken_languages_parsed = df2['spoken_languages'].apply(lambda x: get_list(x, 'iso_639_1'), meta=object)
df2 = df2.drop(columns=['spoken_languages'])
df2 = df2.assign(spoken_languages=spoken_languages_parsed)

# production_companies
production_companies_parsed = df2['production_companies'].apply(lambda x: get_list(x, 'name'), meta=object)
df2 = df2.drop(columns=['production_companies'])
df2 = df2.assign(production_companies=production_companies_parsed)

# production_countries
production_countries_parsed = df2['production_countries'].apply(lambda x: get_list(x, 'name'), meta=object)
df2 = df2.drop(columns=['production_countries'])
df2 = df2.assign(production_countries=production_countries_parsed)

# belongs_to_collection
belongs_to_collection_parsed = df2['belongs_to_collection'].apply(lambda x: get_object(x, 'name'), meta=object)
df2 = df2.drop(columns=['belongs_to_collection'])
df2 = df2.assign(belongs_to_collection=belongs_to_collection_parsed)



# RESUME OF CLEANING DATA PROCESS
print("DataFrame columns: ")  # 15 (antes 24)
print(df2.columns)
print("DataFrame rows (number of): " + str(len(df2.index)))  #43680 (antes 45466)

#### DESCRIPTIVE STATISTICS ####

#print(df)
# Numeric value columns: budget, revenue, runtime, vote average, vote count

"""
# Budget
print("\nBUDGET: ")
print(df2['budget'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df2['budget'].values).compute())))


# Revenue
print("\nREVENUE: ")
print(df2['revenue'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df2['revenue'].values).compute())))

# Runtime
print("\nRUNTIME: ")
print(df2['runtime'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df2['runtime'].values).compute())))

# Vote Average
print("\nVOTE AVERAGE: ")
print(df2['vote_average'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df2['vote_average'].values).compute())))

# Vote Count
print("\nVOTE COUNT: ")
print(df2['vote_count'].describe().compute())
print("skewness: " + str(float(dask_stats.skew(df2['vote_count'].values).compute())))

# String object columns: original_title, original_language, belongs_to_collection
print("\nORIGINAL LANGUAGE: ")
print(df2['original_language'].describe().compute())

print("\nORIGINAL TITLE: ")
print(df2['original_title'].describe().compute())

print("\nBELONGS TO COLLECTION: ")
print(df2['belongs_to_collection'].describe().compute())

# Date type columns: release_date
print("\nRELEASE YEAR: ")
print(df2['release_year'].describe().compute())

# Bool object column: adult
print("\nADULT: ")
print(df2['adult'].describe().compute())

"""

# SQL QUERYS

query1(df2)

#query8(df)





