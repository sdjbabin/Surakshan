
<img src="https://github.com/sdjbabin/Surakshan/assets/137878044/cb9c5572-096a-4598-a9df-9c4772c371c2" alt="image" width="600" height="300" align="center">


## Table of Contents

- [Running the website](#running-the-website)
- [Introduction](#introduction)
- [About](#about)
- [Dashboard Analyzing Crime Data](#dashboard-analyzing-crime-data)
- [Machine Learning Models](#machine-learning-models)
- [Apriori Algorithm](#apriori-algorithm)
- [DBSCAN Clustering](#dbscan-clustering)
- [Deep Learning Model](#deep-learning-model)
- [Chatbot](#chatbot)
- [Conclusion](#conclusion)
- [Contact Us](#contact-us)
  

  ## View Our Website Demo

You can view our website demo [here](https://example.com).

## Running the website

To install and run the project locally, follow these steps:

1. Clone the repository to your local machine:
   ```sh
   git clone https://github.com/sdjbabin/Surakshan.git

2. Navigate to the project directory:
   ```sh
   cd Surakshan/website


3. Install the required dependencies:

   ```sh
   pip install -r requirements.txt

4. Download the datasets from the drive link and save it to the to main folder where the model.py is present
   [View the datasets](https://drive.google.com/drive/u/1/folders/1WZnXpP1B8oKfjV-uN9HjthmQeSNQUMVN)


6. Finally, open run the model.py file
   ```sh
   python model.py

## Introduction

A thorough Crime Prediction Model is presented by Team Sherlocked to predict crime hotspots, trends, and offender characteristics for the best possible law enforcement tactics. The approach provides hotspot mapping, suspect identification, and crime classification by utilizing AI algorithms and advanced analytics. Data visualization and deployment recommendations are made easier with the help of an intuitive interface. Proactive intervention is made possible through anomaly detection, and predicted accuracy is improved through the integration of LSTM, GCN. Database administration, data processing, and visualization are all part of the architecture stack of the model. The technology facilitates community interaction and improves public safety measures by providing real-time crime updates and individualized help through a distinctive chatbot. This sets the stage for a proactive approach to crime prevention.
![image](https://github.com/sdjbabin/Surakshan/assets/137878044/5ede533b-1b81-496f-bae5-6921e4888102)

## Components 

1. **Interactive Dashboard**
   - Criminal Dashboard
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/e0bfabe6-43f7-446d-8b66-023c81b619a7)

   - Victim Dashboard
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/8d8a6a58-5910-489e-9b4a-155a9260cae2)

   - Crime Dashboard
      ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/58084fb2-ab81-412e-8fb9-cafb9b34c76d)

   - Police Beat Dashboard
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/df56f2a9-778e-4f0c-9a25-0c11982ffed2)


2. **Crime Watch**: Animated Crime Hotspots Visualization


   
<img src="https://github.com/sdjbabin/Surakshan/assets/137878044/fbc5c793-1049-42d5-8b07-c4c8bb6bfeea" alt="plz" width="1200" height="600">




3. **Predictive Analytics**
   - Accused-Victim Demographics
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/4b980aca-b4c2-4ac7-ac67-688e6183969d)

   - Crime Time
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/1228848b-a0b2-4fcb-a32d-758d2884b81d)

   - Crime Hotspots
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/aaabb895-48dc-4201-acdf-89741fe570b0)

   - Crime Type
     ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/5a6159f9-cc19-46d1-a5ce-2e048e8d0168)


4. **Beat Optimization**

  ![image](https://github.com/sdjbabin/Surakshan/assets/137878044/1d055723-9d9d-40c9-b310-5fb53be6e400)


    
## Dashboard Analyzing Crime Data

The dashboard feature for analyzing crime data includes four main components:

1. **Criminal Dashboard**: This dashboard provides insights into criminal activities, such as the types of crimes committed, frequency, and trends over time.

2. **Victim Dashboard**: Here, users can analyze victim-related data, including demographics, types of victimization, and geographical distribution.

3. **Crime Dashboard**: This dashboard focuses on overall crime statistics, including crime rates, crime categories, and their spatial and temporal patterns.

4. **Police Beat Dashboard**: Users can explore police beat-related data, such as patrol routes, response times, and crime prevention measures implemented in different areas.


## Machine Learning Models

In this project, we have utilized the Naïve Bayes machine learning model for crime prediction. After evaluating various machine learning algorithms, including decision trees, logistic regression, and support vector machines, we found that the Naïve Bayes model consistently provided superior predictions and accuracy compared to other models.

The Naïve Bayes algorithm is particularly well-suited for this task due to its simplicity, efficiency, and ability to handle large datasets with categorical features. It is based on the principle of Bayes' theorem, which calculates the probability of a hypothesis given the evidence.




## Apriori Algorithm

Our goal of using this model is to find all possible crime frequent patterns regardless of the committed crime type. Hence, we implemented the Apriori algorithm on location and time features and excluded the crime type feature. Additionally, to obtain more frequent patterns we applied constraint-based mining by restricting the extraction process on the frequent patterns having this formula of three specific item sets (Crime Type, Day,Time). Then we selected the optimum choice.
For Karnataka, the minimum support value was 0.008, which corresponds to 277 absolute frequencies.
![image](https://github.com/sdjbabin/Surakshan/assets/137878044/54e70f23-6530-41c7-aa9d-b91cb10850e5)



## DBSCAN Clustering

Clustering of crimes within each of 40 districts was performed for the time periods 2016-18,2019-21,2022-23, and temporal changes in cluster distributions were analyzed.

Define Systemic crime to be a district in which the crime clusters have not changed in geography over time.

![image](https://github.com/sdjbabin/Surakshan/assets/137878044/6b2963b0-0371-448c-9497-d5b441d9e416)

## Deep Learning Model

![image](https://github.com/sdjbabin/Surakshan/assets/137878044/e3751421-a8ad-4ca4-b534-49623f127a6b)
![image](https://github.com/sdjbabin/Surakshan/assets/137878044/2d02a5c6-f151-4f5f-8f36-e71b129e1f69)
![image](https://github.com/sdjbabin/Surakshan/assets/137878044/3aa1ecaf-f14c-4510-8740-d12fadd070a9)

## Chatbot

![image](https://github.com/sdjbabin/Surakshan/assets/137878044/f05fa63b-26f5-499b-89fb-99c6811ae355)

![image](https://github.com/sdjbabin/Surakshan/assets/137878044/518619f1-811f-499e-8faa-9610a819e984)

## Conclusion

In conclusion, the development of an AI-based model for analyzing and predicting crimes in Karnataka holds significant promise for enhancing public safety and law enforcement strategies. By leveraging machine learning algorithms, such as Naïve Bayes, Apriori, DBSCAN clustering, and deep learning models, we have been able to extract valuable insights from spatiotemporal crime data.

The implementation of interactive dashboards, predictive analytics, and beat optimization tools provides law enforcement agencies with actionable intelligence for crime prevention and resource allocation. Furthermore, the visualization of crime hotspots and trends allows for proactive intervention and strategic planning.

Moving forward, continued research and development in the field of crime prediction and analysis will be essential for addressing emerging challenges and evolving criminal behaviors. By leveraging advancements in artificial intelligence, data analytics, and spatial modeling, we can further enhance the effectiveness of law enforcement efforts and contribute to safer communities.




## Contact Us

For further inquiries or collaboration opportunities, please feel free to contact us at [sohamtinni@zohomail.in](mailto:sohamtinni@zohomail.in).
