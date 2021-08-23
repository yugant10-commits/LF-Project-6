from src.function import (
    clean_data,
    start_pipeline,
    extract_keywords,
    remove_col,
    get_similarity,
)
from src.recommendation import recommendations
import pandas as pd
from rake_nltk import Rake
import numpy as np
import streamlit as st


df = pd.read_csv("dataset/imdb_250.csv")

# Applying functions.
clean_df = (
    df.pipe(start_pipeline).pipe(clean_data).pipe(extract_keywords).pipe(remove_col)
)

similarity_score = get_similarity(clean_df)

st.title("Recoomendation of movies")

st.header("Please enter a movie name and get recommendation based on it.")

user_input = st.text_input("", "Logan")

try:
    recommendation = recommendations(user_input, clean_df, similarity_score)
except Exception:
    st.write("Movie not in our database.")

st.subheader("Recommendation for you")
st.write(recommendation)
