import pandas as pd


def recommendations(title, df, cosine_sim):
    """Gives out a list of recommendation given the parameters.

    Parameters
    ----------
    title : [string]
        [a string containing the title of the movie on which to recommend.]
    df : [dataframe]
        [the dataframe on which to look up the movie.]
    cosine_sim : [array]
        [array of similarity score which is used to recommend.]

    Returns
    -------
    [list]
        [returns a list of top 10 recommendation based on the plot of the movie.]
    """

    indices = pd.Series(df.index)
    recommended_movies = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return recommended_movies
