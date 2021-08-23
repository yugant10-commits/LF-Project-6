import pandas as pd
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


def start_pipeline(df):
    """Copies the dataframe and selects the required columns.

    Parameters
    ----------
    df : [pandas dataframe]
        [dataframe to start work on.]

    Returns
    -------
    [pandas dataframe]
        [a new dataframe with selected columns.]
    """
    new_df = df.copy()
    new_df = new_df[["Title", "Genre", "Director", "Actors", "Plot"]]
    return new_df


def clean_data(df):
    """Cleans the dataset

    Parameters
    ----------
    df : [pandas dataframe]
        [dataframe to be cleaned]

    Returns
    -------
    [pandas dataframe]
        [returns a cleaned and processed dataframe.]
    """
    df["Actors"] = df["Actors"].map(lambda x: x.split(",")[:3])

    df["Genre"] = df["Genre"].map(lambda x: x.lower().split(","))

    df["Director"] = df["Director"].map(lambda x: x.split(" "))

    for index, row in df.iterrows():
        row["Actors"] = [x.lower().replace(" ", "") for x in row["Actors"]]
        row["Director"] = "".join(row["Director"]).lower()

    return df


def extract_keywords(df):
    """extracts only the main keywords from the plot.

    Parameters
    ----------
    df : [pandas dataframe]
        [requires a dataframe from which the keywords are extracted.]

    Returns
    -------
    [pandas dataframe]
        [returns a dataframe that as a new column in place of the old one.]
    """
    df["key_words"] = ""

    for index, row in df.iterrows():
        plot = row["Plot"]

        rake = Rake()

        rake.extract_keywords_from_text(plot)

        key_words_dict_scores = rake.get_word_degrees()

        row["key_words"] = list(key_words_dict_scores.keys())

    df.drop(columns=["Plot"], inplace=True)
    df.set_index("Title", inplace=True)

    return df


def remove_col(df):
    """concatenates all the columns into one and removes the others

    Parameters
    ----------
    df : [pandas dataframe]
        [dataframe to be processed]

    Returns
    -------
    [pandas dataframe]
        [returns a datafram which has just one column.]
    """
    df["bag_of_words"] = ""
    columns = df.columns
    for index, row in df.iterrows():
        words = ""
        for col in columns:
            if col != "Director":
                words = words + " ".join(row[col]) + " "
            else:
                words = words + row[col] + " "
        row["bag_of_words"] = words
    df.drop(columns=[col for col in df.columns if col != "bag_of_words"], inplace=True)
    return df


def get_similarity(df):
    """vectorizes the words in the column of the dataframe and computes the similarity.

    Parameters
    ----------
    df : [pandas dataframe]
        [the dataframe which has only one column.]

    Returns
    -------
    [array]
        [an array of similarity scores.]
    """
    count = CountVectorizer()
    count_matrix = count.fit_transform(df["bag_of_words"])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim
