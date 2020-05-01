import pandas as pd
import numpy as np
import os
import warnings

os.chdir("/Users/Madhusudan Prajapati/Desktop/Home-Price")
import pickle
warnings.filterwarnings("ignore")
#Read The Dataset
data=pd.read_csv("Home-Price.csv")
data.info()

#Check the Missing values in the dataset
data.isnull().sum()

#Handling the missing values
data.Bedrooms.median()
data.Bedrooms=data.Bedrooms.fillna(data.Bedrooms.median())
data

#Create the input and output variable
l=['Area','Bedrooms','Old']
X=data[l]
y=data['Price']

#Create the model
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X,y)

pickle.dump(dtc,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))