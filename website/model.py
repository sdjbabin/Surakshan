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

import folium
from folium.plugins import HeatMap
import numpy as np
import requests
# Load the dataset (assuming df is already loaded)
# Replace 'your_dataset.csv' with the actual file path if needed
@app.route('/')
def index():
    
    

    return render_template('index1.html')
   
@app.route('/process_option', methods=['POST'])
def process_option():
    print('hh')
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv('all_merged.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    df1=df[df.District_Name == selected_option]
    response={'items2':(df1['Village_Area_Name'].unique()).tolist()}
    return jsonify(response)

@app.route('/page')
def page():
    return render_template('index.html')

    
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('all_merged.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    data = request.get_json()

   







 
    # Take user input for the district name
    data = request.get_json()

    district_name=data["Area"]
    # Count the occurrences of each village name in the specified district
    acc_caste_counts = df[df.District_Name == district_name]['Caste_x'].value_counts()[0:7]

    # Filter village names that appear more than 120 times (adjust as needed)
    selected_acc_caste = acc_caste_counts.index.tolist()

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()

    # Filter data to include only selected villages
    filtered_df = df[df.District_Name == district_name][df[df.District_Name == district_name]['Caste_x'].isin(selected_acc_caste)]
    outputs=['Caste_x','Profession_x','Sex_x','Caste_y','Profession_y','Sex_y']
    responses=[]
    for out in outputs:
        X = filtered_df[['Month', 'DayOfWeek', 'CrimeGroup_Name','Village_Area_Name']]
        y = label_encoder.fit_transform(filtered_df[out])
    
        # Define column transformer for one-hot encoding categorical features
        categorical_features = ['Month', 'DayOfWeek', 'CrimeGroup_Name','Village_Area_Name']
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




    
        data = request.get_json()
    
        sample_df = pd.DataFrame([data])
        print(data)  # Wrap data in a list to ensure DataFrame creation
    
        probabilities = nb_classifier.predict_proba(sample_df)
        
        probabilities = probabilities[0] 
        if out=='Sex_x' or out=='Sex_y' :
            top_3_indices = np.argsort(probabilities)[::-1][:1]
            top_3_probabilities = probabilities[top_3_indices]
            top_3_castes = label_encoder.inverse_transform(top_3_indices)
        else:
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_3_probabilities = probabilities[top_3_indices]
            top_3_castes = label_encoder.inverse_transform(top_3_indices)
        # Prepare response
        response = []
        for classes, probability in zip(top_3_castes, top_3_probabilities):
            response.append({'classes': classes.upper(), "Probability": probability})
        responses.append(response)
    print(responses)
    return jsonify(responses)

   






if __name__ == '__main__':
    app.run(debug=True)
