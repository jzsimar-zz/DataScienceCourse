###############################CRIME DATA###############################

import pandas as pd
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import	AgglomerativeClustering as aggcluster 

crimes = pd.read_csv("C:\\Users\\jzsim\\Downloads\\crime_data.csv")

#normalization of data so that the data is unit and scale free
def normal(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Appliying normaliztion function on the given university data excluding ID column
newuniv = normal(crimes.iloc[:,1:])
newuniv.describe()

#applying linkage funciton to get the distances point to point and forming clusters
z = linkage(newuniv, method="complete",metric="euclidean")

#plotting dendogram using the z values to define clusters
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
n=5
clustering	= aggcluster(n_clusters=n,linkage='ward',affinity = "euclidean").fit(newuniv) 
clustering.labels_
#assigining defines clusters to a new series
cluster=pd.Series(clustering.labels_)

# creating a  new column and assigning cluster
crimes['clust']=cluster 
#taking column wise and all row data 
crimes = crimes.iloc[:,[5,0,1,2,3,4]]
crimes.head()

# getting aggregate mean of each cluster
crimes.groupby(crimes.clust).mean()

#Inferences
"""
          Murder     Assault   UrbanPop       Rape
clust                                             
0      10.815385  257.384615  76.000000  33.192308
1       4.644444  144.444444  79.222222  18.766667
2       3.091667   76.000000  52.083333  11.833333
3      14.671429  251.285714  54.285714  21.685714
4       7.466667  135.666667  63.444444  18.600000
"""




###############################EAST WEST AIRLINE###############################

#Hierarchical
import pandas as pd
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt
from sklearn.cluster import	AgglomerativeClustering as aggcluster 

airline = pd.read_csv("C:\\Users\\jzsim\\Downloads\\EastWestAirlines.csv")

#normalization of data so that the data is unit and scale free
def normal(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Appliying normaliztion function on the given university data excluding ID column
newairline = normal(airline.iloc[:,1:])
newairline.describe()

#applying linkage funciton to get the distances point to point and forming clusters
z = linkage(newairline, method="complete",metric="euclidean")

#plotting dendogram using the z values to define clusters
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram
n=5
clustering	= aggcluster(n_clusters=n,linkage='ward',affinity = "euclidean").fit(newairline) 
clustering.labels_
#assigining defines clusters to a new series
cluster=pd.Series(clustering.labels_)

# creating a  new column and assigning cluster
airline['Cluster']=cluster 
#taking column wise and all row data 
airline = airline.iloc[:,[12,1,2,3,4,5,6,7,8,9,10,11]]
airline.head()

# getting aggregate mean of each cluster
a=airline.groupby(airline.Cluster).mean()
a

#Inferences
"""
               Balance  Qual_miles  ...  Days_since_enroll    Award?
Cluster                             ...                             
0        116134.226872  367.779736  ...        4699.638767  0.665198
1         46329.336877    9.028618  ...        3769.337694  0.184383
2        134880.892308  393.323077  ...        4599.607692  0.753846
3         68876.581395   23.255814  ...        3968.930233  0.395349
4        129951.388889   65.666667  ...        4488.777778  0.500000
"""

#K Means

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 

airline = pd.read_csv("C:\\Users\\jzsim\\Downloads\\EastWestAirlines.csv")

#normalization of data so that the data is unit and scale free
def normal(i):
    x = (i-i.mean())/(i.std())
    return (x)

# Appliying normaliztion function on the given university data excluding ID column
newairline = normal(airline.iloc[:,1:])
newairline.describe()

###### screw plot or elbow curve ############
k = list(range(2,15))
k

TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(newairline)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(newairline.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,newairline.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=8) 
model.fit(newairline)

model.labels_ # getting the labels of clusters assigned to each row 
cluster= pd.Series(model.labels_)  # converting numpy array into pandas series object 
# creating a  new column and assigning cluster
airline['Cluster']=cluster
#taking column wise and all row data 
airline = airline.iloc[:,[12,1,2,3,4,5,6,7,8,9,10,11]]
airline.head()

# getting aggregate mean of each cluster
a = airline.groupby(airline.Cluster).mean()
a

#Inferences
"""               Balance   Qual_miles  ...  Days_since_enroll    Award?
Cluster                              ...                             
0         43438.468415    41.883989  ...        3699.023441  0.200636
1        117283.428118    60.254964  ...        4893.776807  0.648133
2        138061.400000    78.800000  ...        4613.866667  0.533333
3        190251.952381   458.734694  ...        4673.081633  0.802721
4        119660.491803  5351.065574  ...        3971.491803  0.557377
"""