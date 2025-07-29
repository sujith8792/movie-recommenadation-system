def hybrid_recommendation(user_id, title, movies_df, ratings_df, collab_model, cosine_sim_matrix, n=5):
    # First get content-based recommendations
    content_recs = get_content_recommendations(title, movies_df, cosine_sim_matrix, n*2)
    
    # Then get collaborative filtering predictions for these movies
    predictions = []
    for movie_id in content_recs['movieId']:
        pred = collab_model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top n movies
    top_movie_ids = [x[0] for x in predictions[:n]]
    return movies_df[movies_df['movieId'].isin(top_movie_ids)]