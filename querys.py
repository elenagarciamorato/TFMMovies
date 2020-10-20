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


### Correlation between film's budget and revenue -> Exist correlation
def query1(df):
    # Evaluates an prints correlation matrix
    bud_and_rev = df[['budget', 'revenue']]
    correlation_matrix = bud_and_rev.corr().compute()
    print(correlation_matrix)

    # Represents correlation using a linear regression model (regplot)
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


### 10 Films with lowest budget within the 200 with highest vote_average
### and correlation between budget and vote_average -> Not correlation
def query2(df):
    # Prints a list that contains the 10 films with lowest budget within the 200 with highest vote_average
    # query1 = (df.nsmallest(100, 'budget')).nlargest(10, 'vote_average')
    query = df.nlargest(200, 'vote_average').nsmallest(10, 'budget')
    print(str(query[['vote_average', 'budget', 'original_title']].head(10)))

    # Evaluates an prints correlation matrix
    bud_and_vote_av = df[['budget', 'vote_average']]
    correlation_matrix = bud_and_vote_av.corr().compute()
    print(correlation_matrix)

    # Represents correlation using a linear regression model (regplot)


# 10 oldest films whose original_language is spanish
def query3(df):
    # Print a list that contains the 10 oldest films whose original level is spanish
    query = df[df.original_language == 'es'].nsmallest(10, 'release_year')
    print(str(query[['release_year', 'original_title']].head(10)))

    # Represents released films whose original level is spanish across time using a distribution plot (displot)
    query = df[df.original_language == 'es'].groupby('release_year')
    count_per_year = query['id'].count().compute()

    #seaborn.set(style="whitegrid")
    #f, ax = plt.subplots(figsize=(10, 10))
    #seaborn.despine(f, left=True, bottom=True)
    #seaborn.scatterplot(data=count_per_year.compute())
    #seaborn.histplot(count_per_year.index)
    seaborn.displot(data=count_per_year, x=count_per_year.index)
    plt.show()


# 10 Most voted films with higher vote_average
def query4(df):
    query = df.nlargest(10, ['vote_count', 'vote_average'])
    print(str(query[['vote_count', 'vote_average', 'original_title']].head(10)))
    #query1 = df.nlargest(45000, 'vote_average').nsmallest(15, 'budget')
    #query3 = df.nsmallest(45000, 'vote_average')

    # Representation as


# 15 Films with lowest vote_average within the 200 most voted films
def query5(df):
    query = df.nlargest(200, 'vote_count').nsmallest(15, 'vote_average')
    print(str(query[['vote_count', 'vote_average', 'original_title']].head(15)))


# 5 Years with more released films
def query6(df):
    query = df.groupby('release_year')
    count_per_year = query['id'].count().compute()
    query = count_per_year.nlargest(5)
    print(query.head(5))


# 5 Most popular genres
def query7(df):
    query = df.explode('genres')
    query = query.groupby('genres')
    count_per_genre = query['id'].count().compute()
    query = count_per_genre.nlargest(5)
    print(query.head(5))
    return list(query.index)


# Films by released_year/genre and by revenue/genre
def query8(df):

    # Films whose genres include at least one of the 5 most popular (Drama, Comedy, Thriller, Action, Romance)
    query = df.explode('genres')
    # row_filter = query6['genres'].isin(['Drama', 'Comedy', 'Thriller', 'Action', 'Romance'])
    row_filter = query['genres'].isin(query7(df))

    # Representation released_year/genre
    column_filter = ['release_year', 'genres']
    ages_and_genres = query[row_filter][column_filter].compute()

    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    # group_order = ["Drama", "Comedy", "Thriller", "Action", "Romance"]
    seaborn.violinplot(x="genres", y="release_year", data=ages_and_genres, ax=ax)
    plt.show()

    # Representation revenue/genre
    column_filter = ['revenue', 'genres']
    revenue_and_genres = query6[row_filter][column_filter].compute()

    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    # group_order = ["Drama", "Comedy", "Thriller", "Action", "Romance"]
    seaborn.violinplot(x="genres", y="revenue", data=revenue_and_genres, ax=ax)
    plt.show()

# Films by vote_average/genre
def query8(df):

    # Films whose genres include at least one of the 5 most popular (Drama, Comedy, Thriller, Action, Romance)
    query = df.explode('genres')
    # row_filter = query6['genres'].isin(['Drama', 'Comedy', 'Thriller', 'Action', 'Romance'])
    row_filter = query['genres'].isin(query7(df))

    # Representation released_year/genre
    column_filter = ['vote_average', 'genres']
    vote_and_genres = query[row_filter][column_filter].compute()

    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    seaborn.heatmap(x="genres", y="vote_average", data=vote_and_genres, ax=ax)
    plt.show()

# Films by vote_average/genre (percentage of)
def query9(df):
    query = df.explode('genres')
    query_aux = query.groupby('genres')
    count_per_genre = query_aux['id'].count().compute()
    print(count_per_genre.head(100))


    column_filter = ['vote_average', 'genres', 'id']
    vote_and_genres = query[column_filter].compute()

    vote_average_trun = vote_and_genres['vote_average'].apply(lambda x: int(x))
    vote_and_genres = vote_and_genres.drop(columns=['vote_average'])
    vote_and_genres = vote_and_genres.assign(vote_average=vote_average_trun)


    vote_and_genres = vote_and_genres.groupby(['vote_average', 'genres'])
    print(vote_and_genres.head(100))
    vote_and_genres= vote_and_genres['id'].count()

    heatmap_data = vote_and_genres.reset_index().pivot("vote_average", "genres", "id").fillna(0)
    heatmap_data = heatmap_data / count_per_genre
    #print(heatmap_data)

    # Represents the percentage of films with vote_average/genre using a heatmap
    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    seaborn.heatmap(heatmap_data, ax=ax, cmap='Blues', annot=False, linewidths=1)
    plt.xlabel("Film Genre")
    plt.ylabel("Vote Average")
    plt.show()


