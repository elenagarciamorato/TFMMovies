# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2020/2021

import os

import dask.dataframe as dd
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.impute import KNNImputer
import json
from collections import defaultdict
from functions import *
# Datatype scheme

dtype_scheme ={'budget': np.int64, 'genres': np.object, 'homepage': np.str, 'id': np.int64, 'keywords': np.object,
               'original_language': np.str, 'original_title': np.str, 'overview': np.str, 'popularity': np.float64,
               'production_companies': np.object, 'production_countries': np.object, 'release_date': np.object,
               'revenue': np.int64, 'runtime': np.float64, 'spoken_languages': np.object,  'status': np.object,
               'tagline': np.str, 'title': np.str, 'vote_average': np.float64, 'vote_count': np.int64}

# Change your working directory to where the data resides
os.chdir('/Users/elenagarciamorato/Desktop/TFM')

dfx = dd.read_csv('tmdb_5000_movies.csv', dtype=dtype_scheme)
#df = df.set_index('original_title')
print(dfx)

#peli = dfx.original_title =='Avatar'
#print(str(peli))
#peli=dfx.loc[50].compute()
#prod=peli['production_countries']
#print(prod)


person = '[{"iso_3166_1": "GB", "name": "United Kingdom"}, {"iso_3166_1": "FR", "name": "France"}, {"iso_3166_1": "CA", "name": "Canada"}]'
#
# # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
# print( person_dict)
#
# # Output: ['English', 'French']
person_dict = json.loads(person)
res = defaultdict(list)
{res[key].append(sub[key]) for sub in person_dict for key in sub}
print(str(dict(res)))
print(str(res['name']))

#count_original_languages = df['original_language'].value_counts().compute()
#single_language = list(count_original_languages[count_original_languages == 1].index)
#condition = df['original_language'].isin(single_language)

# prod = json.loads(dfx.production_countries)
# res = defaultdict(list)
# {res[key].append(sub[key]) for sub in prod for key in sub}
# production_countries_masked = dfx['production_countries'].mask(True, res)
# dfx = dfx.drop('production_countries', axis=1)
# dfx = dfx.assign(production_countries=production_countries_masked)

# def str_to_list(x, k):
#     obj = json.loads(x)
#     dictionary = defaultdict(list)
#     {dictionary[key].append(sub[key]) for sub in obj for key in sub}
#     #print(str(dictionary['name']))
#     return dictionary[k]


production_countries_parsed = dfx['production_countries'].apply(lambda x: str_to_list(x, 'name'), meta=object)
dfx = dfx.drop(columns=['production_countries'])
dfx = dfx.assign(production_countries=production_countries_parsed)

print(dfx['production_countries'].head(20))
values = dfx['production_countries'].value_counts().compute()
print(values)
