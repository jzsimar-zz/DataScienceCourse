###########################FOREST FIRES###########################

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

forestfires = pd.read_csv("C:\\Users\\jzsim\\Downloads\\forestfires.csv")
forestfires.head()
forestfires.describe()
forestfires.columns

forestfires.plot.box()

forestfires.plot.bar()

forestfires.isnull().sum()

sns.boxplot(x="size_category",y="FFMC",data=forestfires,palette = "hls")
sns.boxplot(x="size_category",y="DMC",data=forestfires,palette = "hls")
sns.boxplot(x="size_category",y="DC",data=forestfires,palette = "hls")
sns.boxplot(x="size_category",y="ISI",data=forestfires,palette = "hls")
sns.boxplot(x="size_category",y="temp",data=forestfires,palette = "hls")
sns.boxplot(x="FFMC",y="size_category",data=forestfires,palette = "hls")
sns.boxplot(x="DMC",y="size_category",data=forestfires,palette = "hls")


plt.scatter(x=forestfires['FFMC'],y=forestfires['DMC'],color='red')

import seaborn as sns
sns.pairplot(forestfires)

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(forestfires,test_size = 0.3)
test.head()
train_X = train.iloc[:,2:30]
train_X
train_y = train.iloc[:,30]
test_X  = test.iloc[:,2:30]
test_y  = test.iloc[:,30]


model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy = 0.9807692307692307

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 0.9615384615384616

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 0.782051282051282




###########################SALARY DATA###########################

#Not able to get outputs my system crashed when fitting the data

import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train = pd.read_csv("C:\\Users\\jzsim\\Downloads\\SalaryData_Train.csv")
test = pd.read_csv("C:\\Users\\jzsim\\Downloads\\SalaryData_Test.csv")

train1 = pd.get_dummies(train)
test1 = pd.get_dummies(test)

train_X = train1.iloc[:,0:102]
train_X
train_y = train.iloc[:,13]
train_y
test_X  = test1.iloc[:,0:102]
test_X
test_y  = test.iloc[:,13]
test_y

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y) 

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y) 




#Not able to get outputs my system crashed when fitting the data
#so did it only in salary data train
#crashed again


import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train0 = pd.read_csv("C:\\Users\\jzsim\\Downloads\\SalaryData_Train.csv")

train1,test1 = train_test_split(train0,test_size = 0.3)


train2 = pd.get_dummies(train1)
test2 = pd.get_dummies(test1)

train_X = train2.iloc[:,0:102]
train_X
train_y = train1.iloc[:,13]
train_y
test_X  = test2.iloc[:,0:102]
test_X
test_y  = test1.iloc[:,13]
test_y

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y) 

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y) 




#Not able to get outputs my system crashed when fitting the data
#so did it only in salary data test
#crashed again


import pandas as pd 
import numpy as np 
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

test0 = pd.read_csv("C:\\Users\\jzsim\\Downloads\\SalaryData_Test.csv")

train1,test1 = train_test_split(test0,test_size = 0.3)


train2 = pd.get_dummies(train1)
test2 = pd.get_dummies(test1)

train_X = train2.iloc[:,0:102]
train_X
train_y = train1.iloc[:,13]
train_y
test_X  = test2.iloc[:,0:102]
test_X
test_y  = test1.iloc[:,13]
test_y

model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
np.mean(pred_test_linear==test_y) 

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(pred_test_poly==test_y) 

# kernel = rbf # radial base funciton
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y) 

