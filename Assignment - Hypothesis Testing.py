
###########################1###########################

import pandas as pd
import scipy 
from scipy import stats

cutlets=pd.read_csv("C:\\Users\\jzsim\\Downloads\\Cutlets.csv")

#as there are 2 population here comparion with each other.
# checking if both are following normal distribtuion or not.
# doing the same by shapiro test
#H0 : Follworing normal distribution
#Ha : Not Follworing normal distribution
print(stats.shapiro(cutlets['Unit A'])) 
# p Value: 0.3199819028377533
#as P value is greater than 0.05
# P high Null Fly
print(stats.shapiro(cutlets['Unit B'])) 
# p Value: 0.3199819028377533
#as P value is greater than 0.05
# P high Null Fly

#AS BOTH P VALUES ARE GREATER THAN 0.05 P HIGH NULL FLY 
#DATA IS FOLLWOING NORMAL DISTRIBUTION
 
#are external conditions same --> No


# Checking Variances are equal or not
#H0 : VAriances are equal
#Ha : VAriances are not equal

scipy.stats.levene(cutlets['Unit A'], cutlets['Unit B'])
#pvalue=0.4176162212502553
#p high null fly
#Variances are equal

#going for 2 sample test 
#H0 : There is significant difference between size of diameter of cutlets
#Ha : There is no significant difference between size of diameter of cutlets
scipy.stats.ttest_ind(cutlets['Unit A'], cutlets['Unit B'])
#pvalue=0.4722394724599501
# p high null fly
#  Hence There is significant difference between size of diameter of cutlets







###########################2###########################

import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm


labtat=pd.read_csv("C:\\Users\\jzsim\\Downloads\\LabTAT.csv")
labtat


#as there are 4 population here comparion with each other.
# checking if both are following normal distribtuion or not.
# doing the same by shapiro test
#H0 : Follworing normal distribution
#Ha : Not Follworing normal distribution
print(stats.shapiro(labtat.Lab1)) 
# p Value: 0.5506953597068787
# As P value is greater than 0.05
# P high Null Fly
print(stats.shapiro(labtat.Lab2)) 
# p Value: 0.8637524843215942
# As P value is greater than 0.05
# P high Null Fly
print(stats.shapiro(labtat.Lab3)) 
# p Value: 0.4205053448677063
# As P value is greater than 0.05
# P high Null Fly
print(stats.shapiro(labtat.Lab4)) 
# p Value: 0.6618951559066772
# As P value is greater than 0.05
# P high Null Fly

#AS ALL P VALUES ARE GREATER THAN 0.05 P HIGH NULL FLY 
#DATA IS FOLLWOING NORMAL DISTRIBUTION


# Checking Variances are equal or not
#H0 : VAriances are equal
#Ha : VAriances are not equal
scipy.stats.levene(labtat.Lab1,labtat.Lab2,labtat.Lab3,labtat.Lab4)
#pvalue=0.05161343808309816
#p high null fly
#Variances are equal

#Now going for  One - Way Anova test
# H0 : There is difference in turn around time
# Ha : There is no difference in turn around time

from statsmodels.formula.api import ols
labtatmodel=ols('Lab1~Lab2+Lab3+Lab4',data=labtat).fit()
aov_table=sm.stats.anova_lm(labtatmodel,type=2)
print(aov_table)



###########################3###########################

import pandas as pd


buyer=pd.read_csv("C:\\Users\\jzsim\\Downloads\\BuyerRatio.csv")
buyer.drop(['Observed Values'],inplace=True,axis = 1)

#H0: Male Female buyer are similar
#Ha: Male Female buyer are not similar
Chisqres = scipy.stats.chi2_contingency(buyer)
Chi_square=[['','Test Statistic','p-value'],['Buyer',Chisqres[0],Chisqres[1]]]

#pvalue : 0.6603094907091882
#p high null fly

#Hence Male Female buyer are similar



###########################4###########################

import pandas as pd
import scipy 
from scipy import stats

cusorder=pd.read_csv("C:\\Users\\jzsim\\Downloads\\Costomer+OrderForm.csv")
cusorder1=pd.read_csv("C:\\Users\\jzsim\\Downloads\\Costomer+OrderForm.csv")

cusorder.head()
cusorder1.head()

cusorder.Phillippines[cusorder.Phillippines == 'Error Free'] = 0
cusorder.Phillippines[cusorder.Phillippines == 'Defective'] = 1

cusorder.Indonesia[cusorder.Indonesia == 'Error Free'] = 0
cusorder.Indonesia[cusorder.Indonesia == 'Defective'] = 1

cusorder.Malta[cusorder.Malta == 'Error Free'] = 0
cusorder.Malta[cusorder.Malta == 'Defective'] = 1

cusorder.India[cusorder.India == 'Error Free'] = 0
cusorder.India[cusorder.India == 'Defective'] = 1

cus = pd.DataFrame([cusorder['Phillippines'].value_counts(),cusorder.Indonesia.value_counts(),cusorder['Malta'].value_counts(),cusorder['India'].value_counts()])

cus1 = pd.DataFrame([cusorder1['Phillippines'].value_counts(),cusorder1.Indonesia.value_counts(),cusorder1['Malta'].value_counts(),cusorder1['India'].value_counts()])

#H0: Defective Do not Varies by center
#Ha: Defective Varies by center

Chisqres=scipy.stats.chi2_contingency(cus)
Chisqfin=[['','Test Statistic','p-value'],['Sample Data',Chisqres[0],Chisqres[1]]]
#p values : 0.2771020991233144
#p high null fly


Chisqres1=scipy.stats.chi2_contingency(cus1)
Chisqfin1=[['','Test Statistic','p-value'],['Sample Data',Chisqres1[0],Chisqres1[1]]]
#p values : 0.2771020991233144
#p high null fly


#Hence Defective Do Not Varies by center


###########################5###########################

import pandas as pd


faltoon=pd.read_csv("C:\\Users\\jzsim\\Downloads\\Faltoons.csv")

falt = pd.DataFrame([faltoon.Weekdays.value_counts(),faltoon.Weekend.value_counts()])

#H0: Differs by days of week.
#Ha: Do not Differs by days of week.


Chisqres=scipy.stats.chi2_contingency(falt)
Chisqfin=[['','Test Statistic','p-value'],['Sample Data',Chisqres[0],Chisqres[1]]]


#p values : 0.0000854342267020237e-05
#p low null go
#Do not Differs by days of week.