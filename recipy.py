# First, install Flask using: pip install flask
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset
df = pd.read_csv('recipe_recommendation_dataset.csv')

# Preprocess the Preparation Time (convert to numeric for regression)
df['Preparation Time'] = df['Preparation Time'].str.replace(' mins', '').astype(int)

# Adding a 'Cuisine Type' feature (data augmentation)
cuisine_data = {
    'Italian': ['pasta', 'tomato', 'basil'],
    'Indian': ['curry', 'biryani', 'dal'],
    'Mexican': ['taco', 'cilantro', 'avocado'],
    'Chinese': ['soy sauce', 'ginger', 'tofu'],
    # Add more cuisines and their respective ingredients...
}

def assign_cuisine(ingredients):
    for cuisine, ingredients_list in cuisine_data.items():
        if any(ingredient in ingredients for ingredient in ingredients_list):
            return cuisine
    return 'Unknown'

df['Cuisine Type'] = df['Ingredients'].apply(assign_cuisine)

# Split data into training and test sets
X_train, X_test, y_train_time, y_test_time, y_train_recipe, y_test_recipe = train_test_split(
    df['Ingredients'],
    df['Preparation Time'],
    df['Recipe (Instructions)'],
    test_size=0.2,
    random_state=42
)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define pipelines
time_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('scaler', StandardScaler(with_mean=False)),  # Scaling the inputs for regression
    ('regressor', GradientBoostingRegressor())  # Using Gradient Boosting for better performance
])

recipe_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=500))  # Logistic Regression as classifier
])

# Hyperparameter tuning for Preparation Time prediction
param_grid_time = {
    'regressor__n_estimators': [50, 100, 200],  # Number of trees in RandomForest
    'regressor__max_depth': [None, 10, 20, 30],  # Depth of trees
}

# Hyperparameter tuning for Recipe prediction
param_grid_recipe = {
    'classifier__C': [0.1, 1, 10],  # Regularization strength for Logistic Regression
    'classifier__solver': ['liblinear', 'lbfgs']  # Different solvers for Logistic Regression
}

# Using K-Fold CV for both models
time_grid_search = GridSearchCV(time_pipeline, param_grid_time, cv=kf, scoring='neg_mean_absolute_error')
time_grid_search.fit(X_train, y_train_time)

recipe_grid_search = GridSearchCV(recipe_pipeline, param_grid_recipe, cv=kf, scoring='accuracy')
recipe_grid_search.fit(X_train, y_train_recipe)

# Best parameters and models after tuning
best_time_model = time_grid_search.best_estimator_
best_recipe_model = recipe_grid_search.best_estimator_

# Function to predict based on ingredients and return food name, duration, and recipe
def predict_dish(ingredients):
    # Predict preparation time and recipe
    predicted_time = best_time_model.predict([ingredients])
    predicted_recipe = best_recipe_model.predict([ingredients])

    # Retrieve the matching food name from the dataset
    match = df[df['Recipe (Instructions)'] == predicted_recipe[0]]

    if not match.empty:
        food_name = match.iloc[0]['Food Name']
        recipe = match.iloc[0]['Recipe (Instructions)']
    else:
        food_name = "Unknown Dish"
        recipe = "Recipe not found"

    return food_name, predicted_time[0], recipe

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling form input
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ingredients = request.form['ingredients']
        food_name, prep_time, recipe = predict_dish(ingredients)
        return render_template('result.html', food_name=food_name, prep_time=prep_time, recipe=recipe)

if __name__ == '__main__':
    app.run(debug=True)
