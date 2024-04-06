import sklearn
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree 

df=pd.read_csv("irisReady.csv",sep=',',encoding='utf-8')
print(df.head())
#predictive variables
x=df.iloc[:,:3]
#target variables
y=df['species']

print(x.shape,y.shape)

#divise aleatoirement the data to 70% train 30% evaluation
xtrain, xtest, ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1,shuffle=False)
#Create Decision Tree classifier object
treeClassifier=DecisionTreeClassifier()
#Train decision tree classifier
dtree=treeClassifier.fit(xtrain,ytrain)

print(type(dtree))
print(dir(dtree))

print(dtree.get_depth())
print(dtree.criterion)
print(dtree.feature_names_in_)
print(dtree.classes_)
print(dtree.feature_importances_)
print(dtree.max_features_)

print(dir(treeClassifier.tree_))
'''print(treeClassifier.tree_.n_leaves)
print(treeClassifier.tree_.children_left)
print(treeClassifier.tree_.children_right)
print(treeClassifier.tree_.compute_feature_importances)
print(treeClassifier.tree_.compute_node_depths)
print(treeClassifier.tree_.compute_partial_dependence)

tree.plot_tree(dtree, filled=True)
plt.show()'''
print(tree.export_text(treeClassifier))
print(treeClassifier.tree_.children_left)
print(treeClassifier.tree_.children_right)
print(treeClassifier.tree_.compute_node_depths())
