########################Class Assignment########################

import pandas as pd
import numpy as np
diabetes = pd.read_csv("C:\\Xtras\\Assignment Data Science\\Data Set\\Diabetes.csv")

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(diabetes,test_size = 0.3) 
train0 = pd.DataFrame()
test0 = pd.DataFrame()
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,16,2):
    tr = []
    te = []
    neigh = KNC(i)
    print(neigh)
    neigh.fit(train.iloc[:,0:8],train.iloc[:,8])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:8])==train.iloc[:,8])
    print(train_acc)
    test_acc = np.mean(neigh.predict(test.iloc[:,0:8])==test.iloc[:,8])
    print(test_acc)
    tr.append(train_acc)  
    te.append(test_acc)
    train0=train0.append(tr)
    test0=test0.append(te)
        



########################Zoo########################

import pandas as pd
import numpy as np
zoo = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Zoo.csv")       
zoo=zoo.iloc[:,1:18]
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.3) 
train0 = pd.DataFrame()
test0 = pd.DataFrame()
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,26,2):
    tr = []
    te = []
    neigh = KNC(i)
    print(neigh)
    neigh.fit(train.iloc[:,:],train.iloc[:,16])
    train_acc = np.mean(neigh.predict(train.iloc[:,:])==train.iloc[:,16])
    print(train_acc)
    test_acc = np.mean(neigh.predict(test.iloc[:,:])==test.iloc[:,16])
    print(test_acc)
    train0=train0.append([train_acc])
    test0=test0.append([test_acc])
    
#df=pd.DataFrame((train0[0],test0[0]))   



########################Glass########################
    
import pandas as pd
import numpy as np
glass = pd.read_csv("C:\\Users\\jzsim\\Downloads\\glass.csv")       

from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.25) 
train0 = pd.DataFrame()
test0 = pd.DataFrame()
# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC
for i in range(3,26,2):
    tr = []
    te = []
    neigh = KNC(i)
    print(neigh)
    neigh.fit(train.iloc[:,:],train.iloc[:,10])
    train_acc = np.mean(neigh.predict(train.iloc[:,:])==train.iloc[:,10])
    print(train_acc)
    test_acc = np.mean(neigh.predict(test.iloc[:,:])==test.iloc[:,10])
    print(test_acc)
    train0=train0.append([train_acc])
    test0=test0.append([test_acc])

