import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv("irisReady.csv",sep=",",encoding="utf-8")
print(df.head())

#splitting data
#split dataset into training set and test set
#holdout #70% training and 30% test
"""
sklearn.model_selection.train_test_split(
*arrays, test_size=None , train_size=None,
random_state= None, shuffle=True, stratify=None)
"""
#predictive variables
x=df.iloc[:,:3]
#target variable
y=df["species"]
print (x.shape,y.shape)
xtrain , xtest, ytrain , ytest = train_test_split(
    x,y,test_size=0.3,random_state=1)
#