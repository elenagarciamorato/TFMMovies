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
