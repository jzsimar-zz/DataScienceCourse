import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from scipy.cluster.hierarchy import linkage as ln,dendrogram as dd
from sklearn.cluster import	AgglomerativeClustering as aggcluster 
from sklearn.cluster import	KMeans
from scipy.spatial.distance import cdist 


wine = pd.read_csv("C:\\Users\\jzsim\\Downloads\\wine.csv")
wine.describe()

wine.data = wine.iloc[:,1:]
wine.data.head(4)

# Standardization of given data 
winestand = scale(wine.data)

pca = PCA(n_components = 13)
pca_values = pca.fit_transform(winestand)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

# Cumulative variance 
var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="red")

#now as per question
new_df = pd.DataFrame(pca_values[:,:3])

#kmeans
k = list(range(2,14))
k

TWSS = [] # variable for storing total within sum of squares for each kmeans 

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 3 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=3) 
model.fit(new_df)

model.labels_ # getting the labels of clusters assigned to each row 
kmeanscluster= pd.Series(model.labels_)  # converting numpy array into pandas series object 
# creating a  new column and assigning cluster
wine['Kmeans Cluster']=kmeanscluster


#heirarchical

#applying linkage funciton to get the distances point to point and forming clusters
z = ln(new_df, method="complete",metric="euclidean")

#plotting dendogram using the z values to define clusters
plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
dd(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
n=3
clustering	= aggcluster(n_clusters=n,linkage='ward',affinity = "euclidean").fit(new_df) 
clustering.labels_
#assigining defines clusters to a new series
heicluster=pd.Series(clustering.labels_)

# creating a  new column and assigning cluster
wine['Heirarchical clust']=heicluster 


wine.Type[wine.Type == 1] = 0
wine.Type[wine.Type == 2] = 1
wine.Type[wine.Type == 3] = 2

wine = wine.iloc[:,[0,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13]]



