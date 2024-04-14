from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import numpy as np
app = Flask(__name__)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import pandas as pd
import numpy as py
import osmnx as ox
import networkx as nx
import folium
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import folium
from folium.plugins import HeatMap
import numpy as np
import requests
# Load the dataset (assuming df is already loaded)
# Replace 'your_dataset.csv' with the actual file path if needed
@app.route('/')
def index():
    return render_template("index1.html")

@app.route('/process_option', methods=['POST'])
def process_option():
    print('hh')
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    df1=df[df.District_Name == selected_option]
    response={'items2':(df1['Village_Area_Name'].unique()).tolist()}
    return jsonify(response)   

    
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    data = request.get_json()
    

    # Assuming df is your DataFrame and column_name is the name of the specific column
    # Let's say you want to downsample based on the counts of column_name




    # Select relevant features and target variable
    X = df[[ 'Month','DayOfWeek', 'CrimeGroup_Name', 'Village_Area_Name']]  # Features

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['TimeOfDay'])  # Target variable

    # Define column transformer for one-hot encoding categorical features
    categorical_features = ['Month','DayOfWeek', 'CrimeGroup_Name', 'Village_Area_Name']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Initialize and train the Naïve Bayesian classifier
    nb_classifier = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MultinomialNB())
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the classifier to the training data
    nb_classifier.fit(X, y)


    


    # Get input data from request
    data = request.get_json()
    # Convert input data to DataFrame
    sample_df = pd.DataFrame([data])
    print(data)  # Wrap data in a list to ensure DataFrame creation
    # Predict probabilities for each village area
    probabilities = nb_classifier.predict_proba(sample_df)
    
    probabilities = probabilities[0]  # Extract probabilities for the first sample
    # Get the top 3 predicted village areas
    top_3_indices = np.argsort(probabilities)[::-1][:10]
    top_3_probabilities = probabilities[top_3_indices]
    top_3_village_areas = label_encoder.inverse_transform(top_3_indices)
    # Prepare response
    response = []
    for village_area, probability in zip(top_3_village_areas, top_3_probabilities):
        response.append({"Village Area": village_area, "Probability": probability})
    return jsonify(response)

   






if __name__ == '__main__':
    app.run(debug=True,port=49153)
