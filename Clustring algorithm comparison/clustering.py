import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.cluster import KMeans , AgglomerativeClustering , DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

headers = ['status_id','status_type','status_published','num_reactions','num_comments',
          'num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys','1','2','3','4'
]
header = ['status_type','num_reactions','num_comments',
          'num_shares','num_likes','num_loves','num_wows','num_hahas','num_sads','num_angrys'
]

train = pd.read_csv("dataset.csv", header=None,names= headers)
# test = pandas.read_csv("adult.test.csv", header=None , names = header , index_col=False , na_values="?" , skipinitialspace=True)
# dataset = pandas.concat([train,test])
AutoMpg_data = pd.concat([train])
AutoMpg_data.drop(labels=["1", "2", "3", "4",'status_id','status_published'], axis = 1, inplace = True)
AutoMpg_data.info()
for col in header:
    print ("values of %s"%(col))
    print (AutoMpg_data[col].value_counts())
    print (AutoMpg_data[col].count())
    print ('////////////////////////////\n')

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(2, 2, 1)
sns.countplot(AutoMpg_data['num_reactions'], palette = 'pastel')
plt.title('num_reactions', fontsize = 10,color='red')
plt.xlabel('num_reactions', fontsize = 10)
plt.ylabel('count', fontsize = 10)


# checking the Distribution of customers on Weekend
plt.subplot(2, 2, 2)
sns.countplot(AutoMpg_data['num_comments'], palette = 'inferno')
plt.title('num_comments', fontsize = 10,color='red')
plt.xlabel('num_comments', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 3)
sns.countplot(AutoMpg_data['num_likes'], palette = 'husl')
plt.title('num_likes', fontsize = 10,color='red')
plt.xlabel('num_likes', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 4)
sns.countplot(AutoMpg_data['num_loves'], palette = 'Paired')
plt.title('num_loves', fontsize = 10,color='red')
plt.xlabel('num_loves', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(2, 2, 1)
sns.countplot(AutoMpg_data['num_sads'], palette = 'pastel')
plt.title('num_sads', fontsize = 10,color='red')
plt.xlabel('num_sads', fontsize = 10)
plt.ylabel('count', fontsize = 10)


# checking the Distribution of customers on Weekend
plt.subplot(2, 2, 2)
sns.countplot(AutoMpg_data['status_type'], palette = 'inferno')
plt.title('status_type', fontsize = 10,color='red')
plt.xlabel('status_type', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 3)
sns.countplot(AutoMpg_data['num_angrys'], palette = 'husl')
plt.title('num_angrys', fontsize = 10,color='red')
plt.xlabel('num_angrys', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.subplot(2, 2, 4)
sns.countplot(AutoMpg_data['num_shares'], palette = 'Paired')
plt.title('num_shares', fontsize = 10,color='red')
plt.xlabel('num_shares', fontsize = 10)
plt.ylabel('count', fontsize = 10)

plt.show()

plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(2, 2, 1)
sns.countplot(AutoMpg_data['num_wows'], palette = 'pastel')
plt.title('num_wows', fontsize = 10,color='red')
plt.xlabel('num_wows', fontsize = 10)
plt.ylabel('count', fontsize = 10)


# checking the Distribution of customers on Weekend
plt.subplot(2, 2, 2)
sns.countplot(AutoMpg_data['num_hahas'], palette = 'inferno')
plt.title('num_hahas', fontsize = 10,color='red')
plt.xlabel('num_hahas', fontsize = 10)
plt.ylabel('count', fontsize = 10)
AutoMpg_data['status_type'] = AutoMpg_data['status_type'].map({ 'photo':0, 'video':1,'status':2,'link':3})
plt.show()

# AutoMpg_data.drop(labels=["num_likes", "num_shares"], axis = 1, inplace = True)
print(len(AutoMpg_data.columns))
AutoMpg_data = AutoMpg_data.drop(AutoMpg_data.columns[8], axis=1)
print (AutoMpg_data.info())
# print(AutoMpg_data[0].describe())

df = pd.DataFrame(np.random.randn(100, 2))
pca = sklearnPCA(n_components=2)
AutoMpg_data = pd.DataFrame(pca.fit_transform(AutoMpg_data))
plt.scatter(AutoMpg_data[0], AutoMpg_data[1])
plt.show()
Sum_of_squared_distances = []
K = range(1,15)
mms = MinMaxScaler()
mms.fit(AutoMpg_data)
data_transformed = mms.transform(AutoMpg_data)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
N = 7
kmeans = KMeans(n_clusters=N, random_state=1).fit(AutoMpg_data)
highrachial = AgglomerativeClustering(n_clusters=N).fit(AutoMpg_data)
db=DBSCAN(eps=84,min_samples=1).fit(AutoMpg_data)
inertia = kmeans.inertia_
print("inertia kmeans:",inertia)



neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(AutoMpg_data)
distances, indices = nbrs.kneighbors(AutoMpg_data)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
plt.show()

plt.figure(1)
plt.scatter(AutoMpg_data[0], AutoMpg_data[1], c=kmeans.labels_, cmap='viridis' , )
plt.suptitle("KMeans")
plt.show()
plt.figure(2)
plt.suptitle("Highrachial")
plt.scatter(AutoMpg_data[0], AutoMpg_data[1], c=highrachial.labels_, cmap='viridis')
plt.show()
plt.figure(3)
plt.suptitle("DBScan")
plt.scatter(AutoMpg_data[0], AutoMpg_data[1], c=db.labels_, cmap='viridis')
plt.show()