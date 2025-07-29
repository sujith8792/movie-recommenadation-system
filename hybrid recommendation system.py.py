def main():
    st.title("Movie Recommendation System")
    st.write("""
    This system provides movie recommendations based on:
    - Your preferences (content-based)
    - What similar users liked (collaborative filtering)
    - A combination of both (hybrid)
    """)
    
    # Sidebar for user input
    st.sidebar.header("User Preferences")
    
    # Option 1: Search by movie title
    st.sidebar.subheader("Find similar movies")
    movie_title = st.sidebar.selectbox(
        "Select a movie you like:",
        movies['clean_title'].sort_values().unique()
    )
    
    # Option 2: Select preferred genres
    st.sidebar.subheader("Select your favorite genres")
    selected_genres = st.sidebar.multiselect(
        "Choose genres:",
        all_genres,
        default=['Action', 'Adventure', 'Comedy']
    )
    
    # Option 3: Enter user ID for collaborative filtering
    st.sidebar.subheader("Personalized recommendations")
    user_id = st.sidebar.number_input(
        "Enter your user ID (1-1000):",
        min_value=1,
        max_value=1000,
        value=1,
        step=1
    )
    
    # Recommendation type selection
    rec_type = st.sidebar.radio(
        "Recommendation type:",
        ["Content-Based", "Collaborative", "Hybrid"]
    )
    
    # Display recommendations
    st.header("Recommended Movies")
    
    if rec_type == "Content-Based":
        if movie_title:
            try:
                recommendations = get_content_recommendations(movie_title, movies, cosine_sim)
                st.write(f"Movies similar to '{movie_title}':")
                st.dataframe(recommendations[['clean_title', 'genres', 'year']])
            except:
                st.error("Could not find recommendations for this movie.")
        
        if selected_genres:
            genre_recs = get_genre_recommendations(selected_genres, movies)
            st.write(f"Top movies in selected genres ({', '.join(selected_genres)}):")
            st.dataframe(genre_recs[['clean_title', 'genres', 'year']])
    
    elif rec_type == "Collaborative":
        # Get top rated movies by similar users
        st.write("Top recommendations based on users similar to you:")
        # In a real system, we'd implement this properly
        # For demo, we'll just show highly rated movies
        top_movies = ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10)
        top_movies = movies.merge(top_movies, on='movieId')
        st.dataframe(top_movies[['clean_title', 'genres', 'year', 'rating']])
    
    elif rec_type == "Hybrid":
        if movie_title:
            try:
                recommendations = hybrid_recommendation(
                    user_id, movie_title, movies, ratings, collab_model, cosine_sim
                )
                st.write(f"Hybrid recommendations based on '{movie_title}' and users like you:")
                st.dataframe(recommendations[['clean_title', 'genres', 'year']])
            except:
                st.error("Could not generate hybrid recommendations.")
    
    # Display some stats
    st.sidebar.header("Statistics")
    st.sidebar.write(f"Total movies: {len(movies)}")
    st.sidebar.write(f"Total ratings: {len(ratings)}")
    st.sidebar.write(f"Total users: {ratings['userId'].nunique()}")

if __name__ == "__main__":
    main()