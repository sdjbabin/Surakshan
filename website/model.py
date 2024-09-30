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
import webview
import random
from folium.plugins import HeatMap
import numpy as np
import requests
# Load the dataset (assuming df is already loaded)
# Replace 'your_dataset.csv' with the actual file path if needed
@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/criminal')
def criminal():
    
    return render_template('criminal.html')

@app.route('/victim')
def victim():
    
    return render_template('victim.html')
@app.route('/crimewatch')
def crimewatch():
    
    return render_template('crime.html')

@app.route('/demographics')
def demographics():
    
    return render_template('demographics.html')

@app.route('/crimetime')
def crimetime():
    
    return render_template('crime time.html')

@app.route('/crimetype')
def crimetype():
    
    return render_template('type.html')
@app.route('/beatpred')
def beatpred():
    
    return render_template('beat pred.html')

@app.route('/process_option', methods=['POST'])
def process_option():
    data = request.get_json()  # Parse the incoming JSON data
    number = data.get('num')
    datasets=['all_merged.csv','superduperfinal.csv']
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv(datasets[number])
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    df1=df[df.District_Name == selected_option]
    df1['Village_Area_Name']=[x.upper() for x in  df1['Village_Area_Name']]
    df1=df1[~((df1['Village_Area_Name']=='OTHERS')|(df1['Village_Area_Name']=='OTHER'))]
    response={'items2':(df1['Village_Area_Name'].unique()).tolist()}
    return jsonify(response)

@app.route('/process_option1', methods=['POST'])
def process_option1():
    data = request.get_json()  # Parse the incoming JSON data
   
   
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    df1=df[df.Village_Area_Name == selected_option]
   
    response={'items2':(df1['Beat_Name'].unique()).tolist()}
    return jsonify(response)

@app.route('/page')
def page():
    return render_template('index.html')

    
@app.route('/predict1', methods=['POST'])
def predict1():
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

   

@app.route('/predict2', methods=['POST'])
def predict2():
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


@app.route('/predict4', methods=['POST'])
def predict4():
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    data = request.get_json()
    

    # Assuming df is your DataFrame and column_name is the name of the specific column
    # Let's say you want to downsample based on the counts of column_name



    # Get the counts of each value in the specified column
    counts = df['FIR Type'].value_counts()

    # Choose a target count for downsampling, for example, let's say 2
    target_count = 90000

    # Downsample based on the specified column
    downsampled_df = pd.concat([df[df['FIR Type'] == value].sample(target_count) for value, count in counts.items()])

    # Reset index
    downsampled_df.reset_index(drop=True, inplace=True)
    

    # Select relevant features and target variable
    X = downsampled_df[[ 'Month','DayOfWeek', 'TimeOfDay', 'Village_Area_Name']]  # Features

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(downsampled_df['FIR Type'])  # Target variable

    # Define column transformer for one-hot encoding categorical features
    categorical_features = ['Month','DayOfWeek', 'TimeOfDay', 'Village_Area_Name']
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

@app.route('/predict5', methods=['POST'])
def predict5():
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    data = request.get_json()
    district_name=data["Area"]
   
    df=pd.read_csv('superduperfinal.csv')
    
    df['Village_Area_Name']=[x.upper() for x in  df['Village_Area_Name']]
    df=df[~((df['Village_Area_Name']=='OTHERS')|(df['Village_Area_Name']=='OTHER'))]
        
   
   
    # Take user input for the district name
    data = request.get_json()

    district_name=data["Area"]
    # Count the occurrences of each village name in the specified district
    village_counts = df[df.District_Name == district_name]['Village_Area_Name'].value_counts()[0:7]

    # Filter village names that appear more than 120 times (adjust as needed)
    selected_villages = village_counts.index.tolist()

    # Encode the target variable using label encoding
    label_encoder = LabelEncoder()

    # Filter data to include only selected villages
    filtered_df = df[(df['District_Name'] == district_name) & (df['Village_Area_Name'].isin(selected_villages))]

    X = filtered_df[['Month', 'DayOfWeek', 'CrimeGroup_Name']]
    y = label_encoder.fit_transform(filtered_df['Village_Area_Name'])

    # Define column transformer for one-hot encoding categorical features
    categorical_features = ['Month', 'DayOfWeek', 'CrimeGroup_Name']
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

   

    df2 = pd.DataFrame({
        'Village_Area_Name': top_3_village_areas,
        'Probability': top_3_probabilities
    })
    df2['latitude']=[filtered_df[filtered_df['Village_Area_Name']==x]['Latitude'].iloc[0] for x in top_3_village_areas ] 
    df2['longitude']=[filtered_df[filtered_df['Village_Area_Name']==x]['Longitude'].iloc[0] for x in top_3_village_areas ] 
    
    def adjust_duplicates(lst):
        seen = {}
        adjusted_list = []
        
        for value in lst:
            if value in seen:
                # If the value is a duplicate, add a small random value between -0.5 and 0.5
                adjustment = random.uniform(-0.05, 0.05)
                new_value = value + adjustment
                # Ensure the new value is unique by re-checking the seen dictionary
                while new_value in seen:
                    adjustment = random.uniform(-0.05, 0.05)
                    new_value = value + adjustment
                seen[new_value] = True
                adjusted_list.append(new_value)
            else:
                # If the value is not a duplicate, add it to the adjusted list
                seen[value] = True
                adjusted_list.append(value)
        
        return adjusted_list
    print(df2)
    df2['latitude']=adjust_duplicates(df2['latitude'])
    df2['longitude']=adjust_duplicates(df2['longitude'])
    print(df2)
    m = folium.Map(location=[df2['latitude'].mean(), df2['longitude'].mean()], zoom_start=10,tiles='CartoDB dark_matter')

# Prepare data for the heatmap
    heat_data = [[row['latitude'], row['longitude'], row['Probability']] for index, row in df2.iterrows()]
    gradient = {1: 'red', 0.8: 'darkred', 0.6: 'crimson', 0.4: 'firebrick', 0.2: 'darkred'}

# Add the heatmap layer to the map
    HeatMap(heat_data , gradient=gradient).add_to(m)
    for index, row in df2.iterrows():

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            tooltip=row['Village_Area_Name']
        ).add_to(m)
    map_html2 = m.get_root()._repr_html_()

    # Prepare response
    response = []
   
   


    for village_area, probability in zip(top_3_village_areas, top_3_probabilities):
        response.append({"Village Area": village_area, "Probability": probability})
    responses=[response,map_html2,list(top_3_village_areas),list(filtered_df[filtered_df['Village_Area_Name']==top_3_village_areas[0]]['Beat_Name'].unique())]
    
    return jsonify(responses)



if __name__ == '__main__':
    app.run(debug=True)
    
