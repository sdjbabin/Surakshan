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


  


    
@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('superduperfinal.csv')
    df = df[df['CrimeGroup_Name'] != 'OTHERS']
    df = df[df['CrimeGroup_Name'] != 'PROHIBITION OF ACTS']
    data = request.get_json()
    district_name=data["Area"]
   
    df=pd.read_csv('superduperfinal.csv')
    df1=df[df.District_Name==district_name]
    
    # def get_current_location():
    #         try:
    #             url = 'https://ipinfo.io/json'
    #             response = requests.get(url)
    #             response.raise_for_status()  # Raise an exception for HTTP errors
    #             data = response.json()
    #             if 'loc' in data:
    #                 lat, lng = data['loc'].split(',')
    #                 return float(lat), float(lng)
    #             else:
    #                 return None
    #         except Exception as e:
    #             print(f"An error occurred: {e}")
    #             return None

    # current_location = get_current_location()
    
    # print(f"Current Location: Latitude {current_location[0]}, Longitude {current_location[1]}")

    df2=pd.DataFrame(df1.groupby(['Village_Area_Name','Latitude','Longitude']).size().reset_index())
    df2 = df2.rename(columns={0: 'Count'})
    source_point = [16.168543632391266, 75.63233435878055]
    df2=df2.sort_values(by="Count",ascending=False)
    
    hotspots = df2.iloc[0:7,:]
    

    
    target_points = []   # New target points
            
   

    for index, row in hotspots.iterrows():
                point=[row['Latitude'],row['Longitude']]
                target_points.append(point)
            # Create a new Folium map
    m1 = folium.Map(location=[source_point[0], source_point[1]], zoom_start=14)

            # Retrieve the street network within the bounding box of the source point
    G = ox.graph_from_point(source_point, 50000, network_type='drive')

            # Find the nearest nodes in the graph to the source point and each target point
    source_node = ox.distance.nearest_nodes(G, source_point[1], source_point[0])

            # Accumulate the coordinates of each shortest path
    all_shortest_paths_coords = []

    for t in target_points:
            target_node = ox.distance.nearest_nodes(G, t[1], t[0])
            shortest_path = nx.shortest_path(G, source_node, target_node, weight='length')
            length=nx.shortest_path_length(G, source_node, target_node, weight='length')
            if length<50000:
            # Convert shortest path to coordinates
                shortest_path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in shortest_path]
            
            # Append the coordinates to the list
                all_shortest_paths_coords.append(shortest_path_coords)
                folium.Marker(location=(t[0], t[1]), popup='Destination').add_to(m1)

            # folium.Marker(location=(source_point[0], source_point[1])).add_to(m1)
        # Plot all shortest paths on the map
    folium.CircleMarker(
                source_point,
                radius=5, # define how big you want the circle markers to be
                color='yellow',
                fill=True,
                popup='hello',
                fill_color='blue',
                fill_opacity=0.6
            ).add_to(m1)

    for coords in all_shortest_paths_coords:
            folium.PolyLine(coords, color='red').add_to(m1)
    map_html2 = m1.get_root()._repr_html_()











   
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
    filtered_df = df[df.District_Name == district_name][df[df.District_Name == district_name]['Village_Area_Name'].isin(selected_villages)]
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


    # Load the trained model and label encoder
    # nb_classifier = load('nbclassifier_Bagalkot.joblib')
    # label_encoder = load('label_encoder.joblib')


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
    responses=[response,map_html2]
    
    return jsonify(responses)


   






if __name__ == '__main__':
    app.run(debug=True,port=13271)
