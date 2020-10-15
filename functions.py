# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2019/2020

import os

import dask.dataframe as dd
import dask.array as da
import numpy as np
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from dask.array import *
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.impute import KNNImputer
import json
from collections import defaultdict


def get_list(x, k):

    if str(x) == "nan":
        out = "[]"
    else:
        # x = str(x).replace('\\', '')
        # x = str(x).replace('\"', '\'')
        # x = str(x).replace('\'name\': \'\'', '\'name\': \'Unknown\'')
        # x = str(x).replace('{\'', '{\"')
        # x = str(x).replace('\'}', '\"}')
        # x = str(x).replace('\', \'', '\", \"')
        # x = str(x).replace(', \'', ', \"')
        # x = str(x).replace('\': \'', '\": \"')
        # x = str(x).replace('\': ', '\": ')
        # x = str(x).replace('\"\"', '\"')

        mapping = {'\\': '', '\"': '\'', '\'name\': \'\'': '\'name\': \'Unknown\'', '{\'': '{\"', '\'}': '\"}',
                   '\', \'': '\", \"', ', \'': ', \"', '\': \'':  '\": \"', '\': ': '\": ', '\"\"': '\"'}

        for i, j in mapping.items():
            x = x.replace(i, j)

        obj = json.loads(x, strict=False)
        dictionary = defaultdict(list)
        {dictionary[key].append(sub[key]) for sub in obj for key in sub}
        #print(str(dictionary['name']))
        #print(type(dictionary[k]))
        #out = np.array(dictionary[k])
        #out = da.asarray(out)
        out = tuple(dictionary[k])

    return out


def get_object(x, key):

    if str(x) == "nan":
        out = ""
    else:
        mapping = {'\\': '', '\"': '\'', '\'name\': \'\'': '\'name\': \'Unknown\'', 'None': '\'Unknown\'',
                   '{\'': '{\"', '\'}': '\"}','\', \'': '\", \"', ', \'': ', \"', '\': \'':  '\": \"', '\': ': '\": ',
                   '\"\"': '\"'}

        for i, j in mapping.items():
            x = x.replace(i, j)

        obj = json.loads(x, strict=False)
        out = str(obj[key])

    return out


def to_float(x):

    if x.isnumeric():
        return np.float(x)
    else:
        return np.nan

"""""
# Correlation between film's budget and revenue -> Exist correlation
with ProgressBar():
    bud_and_rev = df[['budget', 'revenue']]
    correlation_matrix = bud_and_rev.corr().compute()
print(correlation_matrix)

# Representation as regplot
seaborn.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 10))
seaborn.despine(f, left=True, bottom=True)

# seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, robust=True)
seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, order=2)
plt.ylim(ymin=0)
plt.xlim(xmin=0)
#plt.ylim(ymax=1000000000)
#plt.xlim(xmax=1000000000)
#plt.show()


# 10 films with highest budget produced by Disney
query5 = df.explode('production_companies')
query5 = query5[query5.production_companies == 'Walt Disney Pictures'].nlargest(10, 'budget')
print(str(query5[['budget', 'original_title']].head(10)))

# 10 films with highest budget NON produced by Disney
is_disney = df['production_companies'].apply(lambda x: ('Walt Disney Pictures' in x), meta=object)
query6 = df.assign(is_disney=is_disney)
query6 = query6[query6.is_disney == False]

query6 = query6.nlargest(10, 'budget')
print(str(query6[['budget', 'original_title']].head(10)))

"""""