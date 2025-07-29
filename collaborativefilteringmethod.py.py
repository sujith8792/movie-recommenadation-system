def build_content_model(movies_df):
    # Create TF-IDF matrix for movie titles
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['clean_title'])
    
    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim, tfidf

cosine_sim, tfidf = build_content_model(movies)

def get_content_recommendations(title, movies_df, cosine_sim_matrix, n=5):
    # Get the index of the movie
    idx = movies_df[movies_df['clean_title'].str.lower() == title.lower()].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort movies by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top n similar movies
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices]

def get_genre_recommendations(selected_genres, movies_df, n=5):
    # Filter movies that contain all selected genres
    genre_mask = movies_df[selected_genres].all(axis=1)
    filtered_movies = movies_df[genre_mask]
    
    # Sort by average rating (we'll use ratings count as a proxy here)
    if 'rating' not in filtered_movies.columns:
        # Calculate average rating for each movie
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        filtered_movies = filtered_movies.merge(avg_ratings, on='movieId', how='left')
    
    # Sort by rating and return top n
    return filtered_movies.sort_values('rating', ascending=False).head(n)