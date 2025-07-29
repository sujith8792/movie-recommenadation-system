# movie-recommenadation-system
A Python-based Movie Recommendation System that helps users discover movies based on their preferences using machine learning techniques like content-based filtering and collaborative filtering.

🚀 Features
📌 Recommend movies similar to a user's favorite movie

🧠 Utilizes content-based filtering (based on genres, tags, overview, etc.)

📊 Cleaned and preprocessed movie metadata (titles, genres, overviews)

🔍 Searchable and interactive recommendation interface

🎯 Optimized for performance and accuracy

🧰 Tech Stack
Language: Python 3.x

Libraries Used:

pandas

numpy

scikit-learn

nltk

Flask or Streamlit (for web UI, optional)


How It Works
Data is cleaned and important features like overview, genres, keywords, and cast are combined.

TF-IDF or CountVectorizer is applied to extract feature vectors.

Cosine similarity is used to find and rank similar movies.

The top-N recommendations are displayed based on similarity scores.
