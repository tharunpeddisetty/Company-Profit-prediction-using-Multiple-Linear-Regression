import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')


X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])] ,remainder='passthrough')
X=np.array(ct.fit_transform(X))

#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size=0.2,random_state=0) 


#Training the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),Y_test.reshape(len(y_pred),1)),1))#.reshape is to display the vector vertical instead of default horizontal. axis =0 = vertical cat, axis =1 = horizontal cat :
