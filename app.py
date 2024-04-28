from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the wine dataset
wine_dataset = pd.read_csv('winequality-red.csv')

# Route for home page (EDA and prediction form)
@app.route('/')
def home():
    # EDA Plots
    # Number of values for each quality
    quality_count = wine_dataset['quality'].value_counts().sort_index()

    # Bar plot of volatile acidity vs Quality
    plt.figure(figsize=(8, 6))
    sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
    plt.title('Volatile Acidity vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Volatile Acidity')
    plt.savefig('volatile_acidity_vs_quality.png')  # Save plot as image

    # Bar plot of citric acid vs Quality
    plt.figure(figsize=(8, 6))
    sns.barplot(x='quality', y='citric acid', data=wine_dataset)
    plt.title('Citric Acid vs Quality')
    plt.xlabel('Quality')
    plt.ylabel('Citric Acid')
    plt.savefig('citric_acid_vs_quality.png')  # Save plot as image

    # Convert plots to HTML format to display in the template
    volatile_acidity_plot = '/static/volatile_acidity_vs_quality.png'
    citric_acid_plot = '/static/citric_acid_vs_quality.png'

    return render_template('index.html', quality_count=quality_count,volatile_acidity_plot=volatile_acidity_plot,citric_acid_plot=citric_acid_plot)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Load the train
    # ed Gradient Boosting model
    X = wine_dataset.drop('quality',axis=1)
    Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, Y_train)

    # Get input values from the form
    input_features = [float(x) for x in request.form.values()]
    input_features_np = np.array(input_features).reshape(1, -1)

    # Predict wine quality
    prediction_gb = gb_model.predict(input_features_np)

    # Display prediction result
    if prediction_gb[0] == 1:
        result = 'Good Quality Wine (Gradient Boosting)'
    else:
        result = 'Bad Quality Wine (Gradient Boosting)'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
