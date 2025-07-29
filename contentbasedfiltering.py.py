from textblob import TextBlob

def analyze_sentiment(tags_df):
    # Calculate sentiment polarity for each tag
    tags_df['sentiment'] = tags_df['tag'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return tags_df

tags = analyze_sentiment(tags)

def get_sentiment_recommendations(movie_id, tags_df, movies_df, n=5):
    # Get average sentiment for each movie
    movie_sentiment = tags_df.groupby('movieId')['sentiment'].mean().reset_index()
    
    # Merge with movies
    movie_sentiment = movie_sentiment.merge(movies_df, on='movieId', how='left')
    
    # Sort by sentiment
    return movie_sentiment.sort_values('sentiment', ascending=False).head(n)