    from flask import Flask, request, render_template
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_similarity

    app = Flask(__name__)

    # Load your datasets
    users_df = pd.read_csv(r'C:\Users\dayan\final_project\book\book_dataset.csv')
    movies_df = pd.read_csv(r'C:\Users\dayan\final_project\book\Recommendations_movie_books.csv')
    books_df = pd.read_csv(r'C:\Users\dayan\final_project\book\app.py', on_bad_lines='skip')

    # Collaborative Filtering Function
    def collaborative_filtering_recommendations(user_id, num_suggestions):
        user_movie_ratings = users_df.pivot_table(index='User ID', columns='Movie Preference', values='Rating').fillna(0)
        user_movie_ratings = user_movie_ratings.astype(float)

        user_similarity = cosine_similarity(user_movie_ratings)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

        similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]

        recommendations = pd.Series(dtype=float)
        for similar_user in similar_users:
            similar_user_movies = user_movie_ratings.loc[similar_user]
            recommendations = recommendations.add(similar_user_movies, fill_value=0)

        user_rated_movies = user_movie_ratings.loc[user_id]
        recommendations = recommendations.drop(user_rated_movies[user_rated_movies > 0].index, errors='ignore')

        return recommendations.nlargest(num_suggestions)

    # Web Route
    @app.route('/', methods=['GET', 'POST'])
    def home():
        if request.method == 'POST':
            user_id = request.form.get('user_id', type=int)
            num_suggestions = request.form.get('num_suggestions', type=int)

            # Ensure user_id and num_suggestions are provided
            if user_id is not None and num_suggestions is not None:
                collaborative_movies = collaborative_filtering_recommendations(user_id, num_suggestions)

                return render_template('results.html', movies=collaborative_movies)
        return render_template('index.html')

    if __name__ == '__main__':
        app.run(debug=True)
