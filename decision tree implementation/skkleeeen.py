import sklearn
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
iris = datasets.load_iris()
print(type(iris))
print(iris.__dict__)
print(dir(iris))

print(type(iris.data))
print(type(iris.target))
print(iris.feature_names)
print(iris.target_names)
#creation du data frame
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
target=pd.DataFrame(data=iris.target)
print(target)
df=pd.concat([df,target],axis=1)
df.columns=['sepal.length','sepal.width','petal.length','petal.width','species']
#data description
print(df.info())
print(df.head())
print(df.head(3))
print(df.tail(3))
print(df.tail())
print(df.shape)
df['species']=df['species'].astype('category')
print(df.info())
print(iris.DESCR)
#data exploration
print(df.shape)
print(df.describe())

fig,axs=plt.subplots(2,2)
axs[0,0].boxplot(df["sepal.width"])
axs[0,0].set_title('sepal.width')
axs[0,1].boxplot(df["sepal.length"])
axs[0,1].set_title('sepal.length')
axs[1,0].boxplot(df["petal.width"])
axs[1,0].set_title('petal.width')
axs[1,1].boxplot(df["petal.length"])
axs[1,1].set_title('petal.length')
plt.show()
####################################################################################################################
fig,axs=plt.subplots(2,2)
axs[0,0].hist(df["sepal.width"],bins=20)
axs[0,0].set_title('sepal.width')
axs[0,1].hist(df["sepal.length"],bins=20)
axs[0,1].set_title('sepal.length')
axs[1,0].hist(df["petal.width"],bins=20)
axs[1,0].set_title('petal.width')
axs[1,1].hist(df["petal.length"],bins=20)
axs[1,1].set_title('petal.length')
plt.show()
plt.savefig("histogramme des iris.png")
#####################################################################################################
df.plot.scatter(x='sepal.width',y='sepal.length')
plt.show()
plt.savefig("nuage1.png")
df.plot.scatter(x='petal.width',y='petal.length')
plt.show()
plt.savefig("nuage2.png")
df.plot.scatter(x='sepal.width',y='petal.width')
plt.show()
plt.savefig("nuage3.png")
df.plot.scatter(x='sepal.length',y='petal.length')
plt.show()
plt.savefig("nuage4.png")
df.plot.scatter(x='sepal.width',y='petal.length')
plt.show()
plt.savefig("nuage5.png")
df.plot.scatter(x='sepal.length',y='petal.width')
plt.show()
plt.savefig("nuage6.png")

#coorelation negative : les deux termes (xi-xbar) et (yi-ybar) sont opposé (we7da negative et lautre positive)
#coorelation est entre -1 et 1
# si elle est egale 0 pas de coorealtion (les vaeriables independents , sont differents)
#si elle est entre 0.25 et 0.5 elle est positive faible
#1 ou -1 les variables donnent meme interpretation so we study one of them is enough bcz they give the same information
print(df.iloc[:,0:4].corr())

setosa =df[df.species==0]
versicolor =df[df.species==1]
virginica =df[df.species==2]
fig,axs=plt.subplots(2,2)
fig.set_size_inches(10,7)
axs[0,0].hist(setosa['sepal.length'],bins=15,label="Setosa",facecolor="cyan")
axs[0,0].hist(versicolor['sepal.length'],bins=15,label="Versicolor",facecolor="gray")
axs[0,0].hist(virginica['sepal.length'],bins=15,label="virginica",facecolor="pink")
axs[0,0].set_title('sepal.length')

axs[0,1].hist(setosa['sepal.width'],bins=15,label="Setosa",facecolor="cyan")
axs[0,1].hist(versicolor['sepal.width'],bins=15,label="Versicolor",facecolor="gray")
axs[0,1].hist(virginica['sepal.width'],bins=15,label="virginica",facecolor="pink")
axs[0,1].set_title('sepal.width')

axs[1,0].hist(setosa['petal.length'],bins=15,label="Setosa",facecolor="cyan")
axs[1,0].hist(versicolor['petal.length'],bins=15,label="Versicolor",facecolor="gray")
axs[1,0].hist(virginica['petal.length'],bins=15,label="virginica",facecolor="pink")
axs[1,0].set_title('petal.length')

axs[1,1].hist(setosa['petal.width'],bins=15,label="Setosa",facecolor="cyan")
axs[1,1].hist(versicolor['petal.width'],bins=15,label="Versicolor",facecolor="gray")
axs[1,1].hist(virginica['petal.width'],bins=15,label="virginica",facecolor="pink")
axs[1,1].set_title('petal.width')
plt.show()

Q1=df['sepal.width'].quantile(0.25)
Q3=df['sepal.width'].quantile(0.75)
IQR=Q3-Q1
#identify outliers
threshold=1.5
outliers=df[(df['sepal.width']<Q1 - threshold * IQR)| (df['sepal.width']>Q3 + threshold * IQR)]
print(outliers)
print(outliers.index)
#drop rows contqining outliers
#df=df.drop(outlisers.index)
#print(df.shape)
#replace outliers with median value
#df.loc(outliers.index, 'sepal.width']=df['sepal.width'].median()
#print(df.loc[outliers.index, 'sepal.width'])

#replace outliers with mean value (better than median , more logic , hower inconv is : ecarttyppe bech yos8or ,
#je veux forcer la distribution de donnees , ki yebdew outliers barcha mehich sol behya 
#df.loc(outliers.index, 'sepal.width']=df['sepal.width'].mean()
#print(df.loc[outliers.index, 'sepal.width'])

#replace outliers with upper and lower limit (les vals extreme 9a3dou extreme , inconv : tnjm val extreme mtkounch extreme , tkoun fel79i9a 8alta , val errone
outliersLow=df[df['sepal.width']<Q1 - threshold * IQR]
df.loc[outliersLow.index, 'sepal.width']=Q1 - threshold * IQR
outliersUp=df[df['sepal.width']>Q3 + threshold * IQR]
df.loc[outliersLow.index, 'sepal.width']=Q3 + threshold * IQR
print(df.loc[outliersLow.index, 'sepal.width'])
print(df.loc[outliersUp.index, 'sepal.width'])


#replace outliers with the mean value of the class : a7sen method
#supervised leqning KNN





#petal.width, petal.width 0.96
#petal.length, petal.length 0.87
#-> petql.length
df['species'].replace([0,1,2],['setosa','versicolor','virginica'],inplace=True)
print(df.tail())

df.iloc[:,[0,1,3,4]].to_csv("irisReady.csv",sep=',' , index=False, encoding='utf-8')


