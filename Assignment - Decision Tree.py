##################Fraud Check##################

import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

fraud = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Fraud_check.csv")
fraud.columns
fraud.describe()

fraud["Taxable.Income"]
help(pd.cut)
fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99619], labels = ["Risky", "Good"])

fraudcheck = fraud.drop(columns=["Taxable.Income"])
fraudcheck1 = pd.get_dummies(fraudcheck.drop(columns = ["TaxInc"]))
fraudcheck2 = pd.concat([fraudcheck1, fraudcheck["TaxInc"]], axis = 1)
fraudcheck2

fraudcheck2.TaxInc.value_counts()

y = list(fraudcheck2['City.Population']) 
plt.boxplot(y) 

z = list(fraudcheck2['Work.Experience']) 
plt.boxplot(z) 

import scipy.stats as st
st.pearsonr(fraudcheck2['City.Population'],fraudcheck2['Work.Experience'])

fraudcheck2.isnull().sum()

fraudcheck2.sample()

colsname  = list(fraudcheck2.columns)
predictors = colsname[0:9]
target = colsname[9]

from sklearn.model_selection import train_test_split
train, test = train_test_split(fraudcheck2, test_size = 0.3)
fraudcheck2["TaxInc"].unique()
from sklearn.tree import DecisionTreeClassifier
help(DecisionTreeClassifier)
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(train[predictors],train[target])


preds = model.predict(test[predictors])
preds
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
np.mean(preds==test.TaxInc) 

help(plot_tree)

plt.figure(figsize=(25,10))
a = plot_tree(model, 
              feature_names=predictors, 
              class_names=target, 
              filled=True, 
              rounded=True, 
              fontsize=14)


##################Company Data##################


import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

cdata = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Company_data.csv")
cdata.columns
cdata
cdata.describe()
 
a = list(cdata['Sales']) 
plt.boxplot(a) 

cdata.boxplot(by ='Age', column =['Sales'], grid = False)

cdata["Sales_Values"] = pd.cut(cdata["Sales"], bins = [-1,7.496325,16.27], labels = ["Low", "High"])

cdata1 = cdata.drop(columns=["Sales"])
cdata2 = pd.get_dummies(cdata1.drop(columns = ["Sales_Values"]))
cdata3= pd.concat([cdata2, cdata1["Sales_Values"]], axis = 1)
cdata3

b = list(cdata3['CompPrice']) 
plt.boxplot(b) 
c = list(cdata3['Income']) 
plt.boxplot(c) 
d = list(cdata3['Advertising']) 
plt.boxplot(d) 
e = list(cdata3['Population']) 
plt.boxplot(e) 
f = list(cdata3['Age']) 
plt.boxplot(f) 

cdata3.isnull().sum()
cdata3.plot.bar()
cdata.plot.box()

colsname  = list(cdata3.columns)
predictors = colsname[1:14]
predictors
target = colsname[14]
target



from sklearn.model_selection import train_test_split

help(train_test_split)
train, test = train_test_split(cdata3, test_size = 0.25)
train[predictors]
train[target]
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "entropy")
model.fit(train[predictors],train[target])


preds = model.predict(test[predictors])
preds
pd.Series(preds).value_counts()
pd.crosstab(test[target],preds)
np.mean(preds==test.Sales_Values) 

plt.figure(figsize=(25,10))
a = plot_tree(model, 
              feature_names=predictors, 
              class_names=target, 
              filled=True, 
              rounded=True, 
              fontsize=14)
