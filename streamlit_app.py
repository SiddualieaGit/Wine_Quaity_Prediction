import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the wine dataset
wine_dataset = pd.read_csv('winequality-red.csv')

# Separate features (X) and target (Y)
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Train the Gradient Boosting model
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, Y_train)

# Define the Streamlit app
def main():
    # Set the title and description of the app
    st.title('Wine Quality Prediction')
    st.write('This app predicts whether a wine is of good quality based on its parameters.')

    # Display form for user input
    st.sidebar.header('User Input')
    fixed_acidity = st.sidebar.slider('Fixed Acidity', min_value=4.0, max_value=16.0, value=8.0)
    volatile_acidity = st.sidebar.slider('Volatile Acidity', min_value=0.1, max_value=2.0, value=0.5)
    citric_acid = st.sidebar.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.3)
    residual_sugar = st.sidebar.slider('Residual Sugar', min_value=0.0, max_value=16.0, value=8.0)
    chlorides = st.sidebar.slider('Chlorides', min_value=0.01, max_value=0.6, value=0.08)
    free_sulfur_dioxide = st.sidebar.slider('Free Sulfur Dioxide', min_value=1.0, max_value=72.0, value=30.0)
    total_sulfur_dioxide = st.sidebar.slider('Total Sulfur Dioxide', min_value=6.0, max_value=289.0, value=115.0)
    density = st.sidebar.slider('Density', min_value=0.9900, max_value=1.0030, value=0.9967)
    pH = st.sidebar.slider('pH', min_value=2.0, max_value=5.0, value=3.3)
    sulphates = st.sidebar.slider('Sulphates', min_value=0.3, max_value=2.0, value=0.6)
    alcohol = st.sidebar.slider('Alcohol', min_value=8.0, max_value=15.0, value=10.0)

    # Collect user inputs into a numpy array for prediction
    input_data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1, -1)

    # Make prediction using the trained model
    prediction = gb_model.predict(input_data)

    # Display prediction result
    if prediction[0] == 1:
        st.write('Prediction: Good Quality Wine')
    else:
        st.write('Prediction: Bad Quality Wine')

# Run the Streamlit app
if __name__ == '__main__':
    main()
