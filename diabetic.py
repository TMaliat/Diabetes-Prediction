#importing the dependancies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#Data Collection & Data Analysis -> PIMA Diabetes Dataset

#loading the Diabetic Dataset to a Pandas dataframe
diabetes_dataset= pd.read_csv('diabetes.csv')

#pd.read_csv?
#printing out the first five rows from the dataset
diabetes_dataset.head()
#number of rows and columns of the dataset, to understand its dimension

print(diabetes_dataset.shape)

#Getting the statistical measurements of the data
statistical_summary = diabetes_dataset.describe()
print(statistical_summary)

print(diabetes_dataset['Outcome'].value_counts())

#Label 0: Non Diabetic & Label 1: Diabetic 
print(diabetes_dataset.groupby('Outcome').mean())

#Separating Data Labels
X=diabetes_dataset.drop(columns='Outcome',axis=1)
print(X)
Y=diabetes_dataset['Outcome']
#data standerdization -> important part of data pre-processing
scaler = StandardScaler()
#Learning the mean & standerd deviation of the dataset
scaler.fit(X)
#Normalizing the data [0.1] from learned standerd deviation
standardized_data = scaler.transform(X)
print(standardized_data)
X=standardized_data
print(X)
Y=diabetes_dataset['Outcome']
print(Y)

#Training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2,stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Training the data -> Support Vector Machine 
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)

#How many times our model is doing correct evaluation?
#accuracy score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of the dataset:" )
print(training_data_accuracy)


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Accuracy Score of the training:" ,test_data_accuracy )

#predictive system
input_data = (11,143,94,33,146,36.6,0.254,51)
#Changing this input to numpy array
input = np.asarray(input_data)
#reshape this data as we are predicting for one instance so that model knows that only 1 instance will be there 
input_reshaped = input.reshape(1,-1)
#Now data has to be standardized for the model to predict the outcome

input_standardized = scaler.transform(input_reshaped)
print(input_standardized)

prediction= classifier.predict(input_standardized)
#print(prediction)
if (prediction[0]==0):
    print("Non Diabetic.")
else: 
    print("Diabetic.")
