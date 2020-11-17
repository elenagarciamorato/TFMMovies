# TRABAJO DE FIN DE MÁSTER - TFMMovies
# Elena Garcia-Morato Piñan
# MASTER EN INGENIERÍA EN SISTEMAS DE DECISIÓN
# UNIVERSIDAD REY JUAN CARLOS
# CURSO 2020/2021

import os

import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
from dask.array import stats as dask_stats
from matplotlib import pyplot as plt
import seaborn


### Correlation between film's budget and revenue -> Exist correlation
def query1(df):
    # Evaluates correlation an prints correlation matrix
    bud_and_rev = df[['budget', 'revenue']]
    correlation_matrix = bud_and_rev.corr().compute()
    print(correlation_matrix)

    #print(correlation_matrix.loc['budget', 'revenue'])

    if correlation_matrix.loc['budget', 'revenue'] >= 0.5:

        # Represents correlation using a linear regression model (regplot)
        seaborn.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(10, 10))
        seaborn.despine(f, left=True, bottom=True)

        # seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, robust=True)
        #seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, order=2)
        seaborn.regplot(x="budget", y="revenue", data=bud_and_rev.compute(), ax=ax, order=2).set_yscale('log')
        #seaborn.lmplot(x="budget", y="revenue", data=bud_and_rev.compute(), logx=True)
        #plt.ylim(ymin=0)
        #plt.xlim(xmin=0)
        #plt.ylim(ymax=1000000000)
        #plt.xlim(xmax=1000000000)
        #plt.show(block=False)

    else:
        print("Correlation coefficient isn't high enough to affirm correlation between budget and revenue variables ("
              + str(correlation_matrix.loc['budget', 'revenue']) + ")")



### Correlation between budget and vote_average -> Not correlation
def query2(df):
    # Prints a list that contains the 10 films with lowest budget within the 200 with highest vote_average
    # query1 = (df.nsmallest(100, 'budget')).nlargest(10, 'vote_average')
    #query = df.nlargest(200, 'vote_average').nsmallest(10, 'budget')
    #print(str(query[['vote_average', 'budget', 'original_title']].head(10)))

    # Evaluates correlation and prints correlation matrix
    bud_and_vote_av = df[['budget', 'vote_average']]
    correlation_matrix = bud_and_vote_av.corr().compute()
    print(correlation_matrix)

    # print(correlation_matrix.loc['budget', 'vote_average'])

    if correlation_matrix.loc['budget', 'vote_average'] >= 0.5:

        # Represents correlation using a quadratic regression model (regplot)
        seaborn.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(10, 10))
        seaborn.despine(f, left=True, bottom=True)

        seaborn.regplot(x="budget", y="vote_average", data=bud_and_vote_av.compute(), ax=ax, robust=True)
        # seaborn.regplot(x="budget", y="vote_average", data=bud_and_vote_av.compute(), ax=ax, order=2)
        # seaborn.regplot(x="budget", y="vote_average", data=bud_and_vote_av.compute(), ax=ax, order=2).set_yscale('log')
        # seaborn.lmplot(x="budget", y="vote_average", data=bud_and_vote_av.compute(), logx=True)
        # plt.ylim(ymin=0)
        # plt.xlim(xmin=0)
        # plt.ylim(ymax=1000000000)
        # plt.xlim(xmax=1000000000)
        #plt.show(block=False)

    else:
        print("Correlation coefficient isn't high enough to affirm correlation between budget and vote_average variables ("
              + str(correlation_matrix.loc['budget', 'vote_average']) + ")")


# 10 oldest films whose original_language is spanish -> Imp
def query3(df):
    # Print a list that contains the 10 oldest films whose original level is spanish
    query = df[df.original_language == 'es'].nsmallest(10, 'release_year')
    print(str(query[['release_year', 'original_title']].head(10)))

    # Represents released films whose original language is spanish across time using a distribution plot (displot)
    query = df[df.original_language == 'es'].groupby('release_year')
    count_per_year = query['id'].count().compute()

    #seaborn.set(style="whitegrid")
    #f, ax = plt.subplots(figsize=(10, 10))
    #seaborn.despine(f, left=True, bottom=True)
    #seaborn.scatterplot(data=count_per_year.compute())
    #seaborn.histplot(count_per_year.index)
    seaborn.displot(data=count_per_year, x=count_per_year.index, binwidth=5)
    #plt.show(block=False)
    #plt.close()


# 10 films with higher vote_average within 200 the most voted films -> IMP
def query4(df):
    # Print a list that contains the 10 films with higher vote_average within 200 the most voted films
    query = df.nlargest(200, 'vote_count').nlargest(10, 'vote_average')
    print(str(query[['vote_average', 'vote_count', 'original_title']].head(10)))
    #query1 = df.nlargest(45000, 'vote_average').nsmallest(15, 'budget')
    #query3 = df.nsmallest(45000, 'vote_average')


# 15 Films with lowest vote_average within the 200 most voted films
def query5(df):
    # Print a list that contains the 10 films with lowest vote_average within the 200 most voted films
    query = df.nlargest(200, 'vote_count').nsmallest(10, 'vote_average')
    print(str(query[['vote_count', 'vote_average', 'original_title']].head(10)))


# 5 Most popular film genres
def query6(df):

    # Map elements contained in 'genres' to rows,
    # so that there will be as many rows for each film as genres associated to it
    query = df.explode('genres')

    # Count how many movies are associated with each genre and select the 5 most numerous
    query = query.groupby('genres')
    count_per_genre = query['id'].count().compute()
    query = count_per_genre.nlargest(5)
    #print(query.head(5))

    # Return a list with 5 most popular genres
    return list(query.index)


# Films by released_year/genre -> Imp
def query7(df):

    # Take films (rows) whose genres include at least one of the 5 most popular (Drama, Comedy, Thriller, Action, Romance)
    query = df.explode('genres')
    # row_filter = query6['genres'].isin(['Drama', 'Comedy', 'Thriller', 'Action', 'Romance'])
    pop_genres = query6(df)
    pop_genres.sort()
    row_filter = query['genres'].isin(pop_genres)

    # Take only the columns 'release_year', 'genres'
    column_filter = ['release_year', 'genres']

    # Apply column an row filters to the df
    ages_and_genres = query[row_filter][column_filter].compute()

    # Represents the relationship between release_year and genre using a violinplot
    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    seaborn.violinplot(x="genres", y="release_year", data=ages_and_genres, ax=ax, order=pop_genres)
    #plt.show(block=False)


# Films by revenue/genre
def query8(df):

    # Take films (rows) whose genres include at least one of the 5 most popular (Drama, Comedy, Thriller, Action, Romance)
    query = df.explode('genres')
    # row_filter = query6['genres'].isin(['Drama', 'Comedy', 'Thriller', 'Action', 'Romance'])
    pop_genres = query6(df)
    pop_genres.sort()
    row_filter = query['genres'].isin(pop_genres)

    # Take only the columns 'revenue', 'genres'
    column_filter = ['revenue', 'genres']

    # Apply column an row filters to the df
    revenue_and_genres = query[row_filter][column_filter].compute()

    # Represents the relationship between revenue and genre using a violinplot
    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    seaborn.violinplot(x="genres", y="revenue", data=revenue_and_genres, ax=ax, order=pop_genres)
    #plt.show(block=False)


# Films by vote_average/genre (percentage of) ->Imp
def query9(df):

    # Count number of films of each genre
    query = df.explode('genres')
    query_aux = query.groupby('genres')
    count_per_genre = query_aux['id'].count().compute()
    # print(count_per_genre.head(100))

    # Take only the columns 'vote_average', 'genres', 'id' of the df
    column_filter = ['vote_average', 'genres', 'id']
    vote_and_genres = query[column_filter].compute()

    # Truncate value of vote average
    vote_average_trun = vote_and_genres['vote_average'].apply(lambda x: int(x))
    vote_and_genres = vote_and_genres.drop(columns=['vote_average'])
    vote_and_genres = vote_and_genres.assign(vote_average=vote_average_trun)


    # Group films by vote average and genres
    vote_and_genres = vote_and_genres.groupby(['vote_average', 'genres'])
    #print(vote_and_genres.head(100))

    # Count number of films in each group
    vote_and_genres = vote_and_genres['id'].count()
    #print(vote_and_genres.head(100))

    # Represents the relationship between vote_average and genre using a heatmap
    heatmap_data = vote_and_genres.reset_index().pivot("vote_average", "genres", "id").fillna(0)
    heatmap_data_pct = heatmap_data / count_per_genre
    #print(heatmap_data_pct)

    seaborn.set(style="whitegrid")
    f, ax = plt.subplots(figsize=(10, 10))
    seaborn.despine(f, left=True, bottom=True)
    seaborn.heatmap(heatmap_data_pct, ax=ax, cmap='Blues', annot=False, linewidths=1)
    plt.xlabel("Film Genre")
    plt.ylabel("Vote Average")
    #plt.show(block=False)




# 5 Years with more released films
""""
def query9(df):
    query = df.groupby('release_year')
    count_per_year = query['id'].count().compute()
    query = count_per_year.nlargest(5)
    print(query.head(5))
"""
