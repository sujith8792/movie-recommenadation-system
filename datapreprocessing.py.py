def build_collaborative_model(ratings_df):
    # Define the reader
    reader = Reader(rating_scale=(0.5, 5))
    
    # Load the data
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    
    # Split the data
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Build the model (user-based collaborative filtering)
    sim_options = {
        'name': 'cosine',
        'user_based': True
    }
    model = KNNBasic(sim_options=sim_options)
    model.fit(trainset)
    
    return model

collab_model = build_collaborative_model(ratings)