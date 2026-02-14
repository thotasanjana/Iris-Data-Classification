# Iris-Data-Classification


1️⃣ Project Title

Iris Flower Classification using K-Nearest Neighbors (KNN) and Gaussian Naive Bayes (GNB)

2️⃣ Objective

The objective of this project is to classify iris flowers into three species:

Setosa

Versicolor

Virginica

using two machine learning algorithms implemented from scratch (without using sklearn):

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes (GNB)

The system also provides a web interface to input flower measurements and predict the species.

3️⃣ Dataset Used

The project uses the Iris Dataset, originally introduced by:

Ronald Fisher

The dataset contains:

150 samples

4 features:

Sepal Length (SL)

Sepal Width (SW)

Petal Length (PL)

Petal Width (PW)

3 classes:

Setosa

Versicolor

Virginica

4️⃣ Technologies Used

Python

NumPy

Pandas

Flask (for web interface)

HTML & CSS

JSON (for storing performance reports)

Pickle (for saving trained models)

5️⃣ Project Workflow
Step 1: Data Preprocessing

Loaded Iris dataset using pandas.

Converted species names into numeric labels.

Manually split data into:

80% Training

20% Testing

Step 2: KNN Algorithm (From Scratch)

KNN works by:

Calculating Euclidean distance between test point and all training points.

Selecting K nearest neighbors.

Assigning the majority class among neighbors.

Performance metrics calculated:

Accuracy

Confusion Matrix

Precision

Recall

F1-Score

Model saved as:

KNN.pkl


Reports saved as:

KNN_train.json
KNN_test.json

Step 3: Gaussian Naive Bayes (From Scratch)

Naive Bayes uses:

Bayes Theorem:

P(C∣X)=P(X)P(X∣C)/P(C)​
	​


Assumptions:

Features follow Gaussian distribution.

Features are conditionally independent.

Performance metrics calculated:

Accuracy

Confusion Matrix

Precision

Recall

F1-Score

Model saved as:

NB.pkl


Reports saved as:

NB_train.json
NB_test.json

6️⃣ Web Application

A Flask web application was developed with:

Input fields for SL, SW, PL, PW

Algorithm selection (KNN or NB)

Predict button

Displays predicted class

Workflow:

User enters measurements.

Selects algorithm.

Backend loads saved model.

Prediction displayed on webpage.

7️⃣ Results

Both KNN and Naive Bayes achieved high accuracy on the Iris dataset.

Typical Accuracy:

KNN: ~95–100%

Naive Bayes: ~93–97%

The models successfully classify flower species based on input measurements.

8️⃣ Conclusion

This project demonstrates:

Implementation of ML algorithms from scratch.

Understanding of classification techniques.

Manual calculation of performance metrics.

Integration of ML models with a web application.

It provides both theoretical understanding and practical implementation of machine learning concepts.

9️⃣ Future Enhancements

Add probability display for Naive Bayes.

Add graphical confusion matrix.

Deploy application to cloud platform.

Add more datasets for classification.
