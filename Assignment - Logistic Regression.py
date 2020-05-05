###################BANK DATA###################

import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression

bank = pd.read_csv("C:\\Users\\jzsim\\Downloads\\bank-full.csv")
 
#splitting the single column string to various strings by using split() function
bank[["Age","Job","Marital","Education","Default","Balance","Housing","Loan","Contact","Day","Month","Duration","Campaign","Pdays","Previous","Poutcome","Y"]] = bank['age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'].str.split(";",expand=True) 
bank.columns
#dropping the original single column with all the values
bank.drop(['age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'],inplace=True,axis = 1)
bank.head()
bank.columns
#now droping the categorical columns which do not provide relevant information of input
bank.drop(['Job','Day', 'Education','Contact','Month','Poutcome'],inplace=True,axis = 1)

#now changing categorical data to discrete binary data
bank.Marital[bank.Marital == '"single"'] = 0
bank.Marital[bank.Marital == '"divorced"'] = 0
bank.Marital[bank.Marital == '"married"'] = 1

bank.Default[bank.Default == '"no"'] = 0
bank.Default[bank.Default == '"yes"'] = 1

bank.Housing[bank.Housing == '"no"'] = 0
bank.Housing[bank.Housing == '"yes"'] = 1

bank.Loan[bank.Loan == '"no"'] = 0
bank.Loan[bank.Loan == '"yes"'] = 1

bank.Y[bank.Y == '"no"'] = 0
bank.Y[bank.Y == '"yes"'] = 1

#checking for null values
bank.isnull().sum()

bank.columns
bank.head()

#forming table at once but hanged my pc
#import seaborn as sns
#sns.pairplot(bank)

#forming crosstable and plotting results 
pd.crosstab(bank.Y,bank.Age).plot(kind="bar")
pd.crosstab(bank.Y,bank.Marital).plot(kind="bar")
pd.crosstab(bank.Y,bank.Balance).plot(kind="bar")
pd.crosstab(bank.Y,bank.Housing).plot(kind="bar")
pd.crosstab(bank.Y,bank.Loan).plot(kind="bar")
pd.crosstab(bank.Age,bank.Loan).plot(kind="bar")

#taking count of various variables in dataframe
sb.countplot(x="Y",data=bank)
sb.countplot(x="Age",data=bank)
sb.countplot(x="Duration",data=bank)
sb.countplot(x="Balance",data=bank)
sb.countplot(x="Pdays",data=bank)
sb.countplot(x="Marital",data=bank)
sb.countplot(x="Default",data=bank)

#forming boxplot to check for mean and outliers
sb.boxplot(data = bank,orient = "v")
sb.boxplot(x="Y",y="Age",data=bank,palette = "hls")
sb.boxplot(x="Y",y="Balance",data=bank,palette = "hls")
sb.boxplot(x="Y",y="Marital",data=bank,palette = "hls")
sb.boxplot(x="Y",y="Loan",data=bank,palette = "hls")
sb.boxplot(x="Y",y="Duration",data=bank,palette = "hls")
sb.boxplot(x="Y",y="Campaign",data=bank,palette = "hls")

#forming model by making different dataframe by splitting inpur from output variable
#input
X = bank.iloc[:,[0,1,2,3,4,5,6,7,8,9]]
#output
Y = bank.iloc[:,-1]
Y=Y.astype('int')
#model
lr = LogisticRegression()
lr.fit(X,Y)
#model coefficient for plotting line
lr.coef_ 
#model probailities for each value point for chance of happening or not
lr.predict_proba (X) 
#if it is greater than 0.50 then chances are possible of happening 

#predicted values from x for possible values of y
Ynewpred = lr.predict(X)
bank["Y Predicted"] = Ynewpred# making a new column with these values in bank dataframe

YProbability = pd.DataFrame(lr.predict_proba(X.iloc[:,:]))#making a data frame from predicted values of y from using x variables
banknew = pd.concat([bank,YProbability],axis=1)#making new data frame with these y probability values

#forming confusion matrix
from sklearn.metrics import confusion_matrix as cf
cf1 = cf(Y,Ynewpred)#between y original and y predicted by the model
print (cf1)
#     y       n
#y [39162   760]
#n [ 4226  1063]]
# accuracy = 39162+1063/39612+760+4226+1063 == 0.88
#(39162+1063)/(39612+760+4226+1063)

from sklearn.metrics import roc_curve as rocc
from sklearn.metrics import roc_auc_score as rocarea

fpr ,tpr ,thresholdasd = rocc(Y,Ynewpred)
auc = rocarea(Y,Ynewpred)

import matplotlib.pyplot as plt
plt.plot(fpr , tpr , color='red' ,label='ROC')





###################Credit Card###################










import pandas as pd
import seaborn as sb
from sklearn.linear_model import LogisticRegression

cc = pd.read_csv("C:\\Users\\jzsim\\Downloads\\creditcard.csv")
 
cc.columns
#dropping the single column with index values
cc.drop(['Unnamed: 0'],inplace=True,axis = 1)
cc.head(30)
cc.columns

#now changing categorical data to discrete binary data
cc.card[cc.card == 'no'] = 0
cc.card[cc.card == 'yes'] = 1

cc.owner[cc.owner =='no'] = 0
cc.owner[cc.owner =='yes'] = 1

cc.selfemp[cc.selfemp =='no'] = 0
cc.selfemp[cc.selfemp =='yes'] = 1

#checking for null values
cc.isnull().sum()

#forming table at once but hanged my pc
import seaborn as sns
sns.pairplot(cc)

#forming crosstable and plotting results 
pd.crosstab(cc.card,cc.age).plot(kind="bar")
pd.crosstab(cc.card,cc.income).plot(kind="bar")
pd.crosstab(cc.card,cc.share).plot(kind="bar")
pd.crosstab(cc.card,cc.months).plot(kind="bar")
pd.crosstab(cc.card,cc.majorcards).plot(kind="bar")
pd.crosstab(cc.dependents,cc.active).plot(kind="bar")

#taking count of various variables in dataframe
sb.countplot(x="card",data=cc)
sb.countplot(x="reports",data=cc)
sb.countplot(x="age",data=cc)
sb.countplot(x="income",data=cc)
sb.countplot(x="share",data=cc)
sb.countplot(x="expenditure",data=cc)
sb.countplot(x="owner",data=cc)
sb.countplot(x="dependents",data=cc)
sb.countplot(x="months",data=cc)


#forming boxplot to check for mean and outliers
sb.boxplot(data = bank,orient = "v")
sb.boxplot(x="card",y="age",data=cc,palette = "hls")
sb.boxplot(x="card",y="income",data=cc,palette = "hls")
sb.boxplot(x="card",y="share",data=cc,palette = "hls")
sb.boxplot(x="card",y="months",data=cc,palette = "hls")
sb.boxplot(x="card",y="majorcards",data=cc,palette = "hls")
sb.boxplot(x="age",y="active",data=cc,palette = "hls")

#forming model by making different dataframe by splitting inpur from output variable
#input
X = cc.iloc[:,[1,2,3,4,5,6,7,8,9,10,11]]

#output
Y = cc.iloc[:,0]
Y=Y.astype('int')

#model
lr = LogisticRegression()
lr.fit(X,Y)
#model coefficient for plotting line
lr.coef_ 
#model probailities for each value point for chance of happening or not
lr.predict_proba (X) 
#if it is greater than 0.50 then chances are possible of happening 

#predicted values from x for possible values of card
cardnewpred = lr.predict(X)
cc["Card Predicted"] = cardnewpred# making a new column with these values in bank dataframe

CardProbability = pd.DataFrame(lr.predict_proba(X.iloc[:,:]))#making a data frame from predicted values of card from using x variables
banknew = pd.concat([cc,CardProbability],axis=1)#making new data frame with these card probability values

#forming confusion matrix
from sklearn.metrics import confusion_matrix as cf
cf = cf(Y,cardnewpred)#between y original and card predicted values by the model
print (cf)
#     y       n
#y [ 295      1]
#n [ 23     1000]]
# accuracy = 295+1000/295+1000+23+1 == 0.98
#(295+1000)/(295+1000+23+1)


from sklearn.metrics import roc_curve as rocc
from sklearn.metrics import roc_auc_score as rocarea

fpr ,tpr ,thresholdasd = rocc(Y,cardnewpred)
auc = rocarea(Y,cardnewpred)

import matplotlib.pyplot as plt
plt.plot(fpr , tpr , color='red' ,label='ROC')


