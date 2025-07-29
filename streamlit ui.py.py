import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import streamlit as st
import os
import zipfile
import requests

# Download and extract the dataset
def download_dataset():
    url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    if not os.path.exists("ml-25m"):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        with open("ml-25m.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        
        with zipfile.ZipFile("ml-25m.zip", "r") as zip_ref:
            zip_ref.extractall(".")
        print("Dataset downloaded and extracted.")
    else:
        print("Dataset already exists.")

download_dataset()

# Load the data
movies = pd.read_csv("ml-25m/movies.csv")
ratings = pd.read_csv("ml-25m/ratings.csv")
tags = pd.read_csv("ml-25m/tags.csv")

# Preprocess movies data
def preprocess_movies(movies_df):
    # Extract year from title
    movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')
    movies_df['year'] = pd.to_numeric(movies_df['year'], errors='coerce')
    
    # Clean title
    movies_df['clean_title'] = movies_df['title'].str.replace(r'\(\d{4}\)', '').str.strip()
    
    return movies_df

movies = preprocess_movies(movies)

# Create genre matrix for content-based filtering
def create_genre_matrix(movies_df):
    # Get all unique genres
    genres = set()
    for genre_list in movies_df['genres'].str.split('|'):
        genres.update(genre_list)
    
    # Create binary columns for each genre
    for genre in genres:
        movies_df[genre] = movies_df['genres'].str.contains(genre).astype(int)
    
    return movies_df, list(genres)

movies, all_genres = create_genre_matrix(movies)

# Prepare ratings data
def prepare_ratings(ratings_df):
    # We'll sample the data to make it more manageable for this demo
    ratings_df = ratings_df.sample(frac=0.1, random_state=42)
    return ratings_df

ratings = prepare_ratings(ratings)