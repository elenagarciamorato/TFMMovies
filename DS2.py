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

dtype_scheme = {'budget': np.object, 'genres': np.object, 'homepage': np.str, 'id': np.object,
                'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.object,
                 'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
                 'revenue': np.float64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
                 'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.float64,
                 'adult': np.object, 'belongs_to_collection': np.object, 'imdb_id': np.object, 'poster_path': np.object,
                'video': np.object}

# dtype_scheme2 = {'adult': np.object, 'belongs_to_collection': np.object, 'budget': np.object, 'genres': np.object, 'homepage': np.str,                 'id': np.object, 'imdb_id': np.object, 'original_language': np.str, 'original_title': np.str, 'overview': np.str,
#                  'popularity': np.object, 'poster_path': np.object, 'production_companies': np.object,
#                  'production_countries': np.object, 'release_date': np.object, 'revenue': np.object, 'runtime': np.object,
#                  'spoken_languages': np.object,  'status': np.object, 'tagline': np.str, 'title': np.str, 'video': np.object,
#                  'vote_average': np.object, 'vote_count': np.object}
#
# dtype_scheme3 = {'budget': np.int64, 'genres': np.object, 'homepage': np.str, 'id': np.int64,
#                  'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
#                 'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
#                  'revenue': np.int64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
#                  'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.int64,
#                 'adult': bool, 'belongs_to_collection': np.object, 'imdb_id': np.object, 'poster_path': np.object,
#                  'video': bool}

# Change your working directory to where the data resides
os.chdir('/Users/elenagarciamorato/Desktop/TFM')

df2 = dd.read_csv('movies_metadata.csv', dtype = dtype_scheme, error_bad_lines=False, sep=',', na_values=[' ', "NA", '[]'])
#df2 = dd.read_csv('prueba3.csv', dtype = dtype_scheme, error_bad_lines=False, sep=';', na_filter=False)
#df2 = dd.read_csv('movies_metadata.csv',dtype = dtype_scheme, error_bad_lines=False)
#df = df.set_index('original_title')
print(df2)

######## DATA CLEANING #########
# columns and number of rows
print("DataFrame columns: ")
print(df2.columns)
#print("DataFrame rows (number of): " + str(len(df2)))

print(df2.head(5))


## NULL VALUES

# null values count
missing_values = df2.isnull().sum()
# null values %
missing_count = ((missing_values / df2.index.size) * 100)

# compute
with ProgressBar():
    missing_count_pct = missing_count.compute()
print(missing_count_pct)
#print(df.head())
