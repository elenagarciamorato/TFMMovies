# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2020/2021

import os

import dask.dataframe as dd
import dask.array as da
import numpy as np
import seaborn
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from dask.array import *
from matplotlib import pyplot as plt
from datetime import datetime
from sklearn.impute import KNNImputer
import json
from collections import defaultdict
from scipy.stats import ks_2samp


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


def hide_labels(f):
    for axx in f.get_axes():
        for label in axx.get_xticklabels(which="both"):
            label.set_visible(False)
        axx.get_xaxis().get_offset_text().set_visible(False)
        axx.set_xlabel("")


def ks_twosampletest(a, b, var):

    test = ks_2samp(a.astype(str), b.astype(str))
    #print(str(test))

    c = 1.138

    if test[0] > c*sqrt((len(a) + len(b)) / (len(a) * len(b))):
        print("Para la variable " + var + " se rechaza la hipotesis nula que afirma que ambas muestras provienen de la misma distribución")
    else:
        print(
            "Para la variable " + var + " NO puede rechazarse la hipotesis nula que afirma que ambas muestras provienen de la misma distribución")


def missing_at_random_test(df, variables):
    aux = df.explode('genres').explode('spoken_languages').explode('production_countries')

    for x in variables:

        filter = aux[x].isnull()
        null_df = aux[filter].compute()
        notnull_df = aux[filter == False].compute()

        # Categorial variables
        seaborn.set(style="whitegrid")
        f, ax = plt.subplots(4, 2, constrained_layout=True)

        hide_labels(f)

        f.suptitle('Missing vs Not Missing ' + str(x) + ' value', fontsize=18)
        seaborn.despine(f, left=True, bottom=True)

        categorical_v = ['original_language', 'genres', 'spoken_languages', 'production_countries']

        for i in range(0, len(categorical_v)):
            var = categorical_v[i]
            seaborn.countplot(null_df[var], ax=ax[i, 0], order=null_df[var].value_counts().sort_values().index)
            seaborn.countplot(notnull_df[var], ax=ax[i, 1], order=notnull_df[var].value_counts().sort_values().index)

            # We use the Kolmogorov–Smirnov test to compare the equality of both distributions (two-sample K–S test).
            ks_twosampletest(null_df[var].astype(str), notnull_df[var].astype(str), var)

        plt.show()

        # Numerical variables
        seaborn.set(style="whitegrid")
        f, ax = plt.subplots(5, 2, constrained_layout=True)
        f.suptitle('Missing vs Not Missing ' + str(x) + ' value', fontsize=18)
        seaborn.despine(f, left=True, bottom=True)

        numerical_v = ['vote_average', 'vote_count', 'release_year', 'runtime', 'budget', 'revenue']
        numerical_v.remove(x)

        for i in range(0, len(numerical_v)):
            var = numerical_v[i]
            # print(str(var))
            seaborn.distplot(null_df[var], ax=ax[i, 0], axlabel=False)
            seaborn.distplot(notnull_df[var], ax=ax[i, 1], axlabel=False)

            # We use the Kolmogorov–Smirnov test to compare the equality of both distributions (two-sample K–S test).
            ks_twosampletest(null_df[var].astype(str), notnull_df[var].astype(str), var)

        plt.show()

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