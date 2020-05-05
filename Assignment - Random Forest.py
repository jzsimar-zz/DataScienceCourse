##################Fraud Check##################

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as AS
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

fraudcheck2.plot.box()

colsname  = list(fraudcheck2.columns)
predictors = colsname[0:9]
target = colsname[9]

X = fraudcheck2[predictors]
Y = fraudcheck2[target]

rfc = RFC(n_jobs=2,oob_score=True,n_estimators=1000,criterion="entropy")

np.shape(fraudcheck2) 

rfc.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rfc.oob_score_

rfc.predict(X)
fraudcheck2['rfc_pred'] = rfc.predict(X)
print('Model accuracy score: {0:0.4f}'. format(AS(fraudcheck2['TaxInc'],fraudcheck2['rfc_pred'])))




##################Company Data##################


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as AS

cdata = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Company_data.csv")
cdata.columns
cdata
cdata.describe()
 
a = list(cdata['Sales']) 
plt.boxplot(a) 

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


X = cdata3[predictors]
Y = cdata3[target]

rfc = RFC(n_jobs=4,oob_score=True,n_estimators=100,criterion="entropy")

np.shape(cdata3) 

rfc.fit(X,Y) # Fitting RandomForestClassifier model from sklearn.ensemble 
rfc.oob_score_

rfc.predict(X)
cdata3['rfc_pred'] = rfc.predict(X)

print('Model accuracy score: {0:0.4f}'. format(AS(cdata3['Sales_Values'],cdata3['rfc_pred'])))

