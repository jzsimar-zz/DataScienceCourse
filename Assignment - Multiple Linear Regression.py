#############Toyota Corolla#############


import pandas as pd 
toyocars = pd.read_csv("C:\\Users\\jzsim\\Downloads\\ToyotaCorolla.csv",encoding='latin1')
type(toyocars)
toyocars.head() 

#selecting specified columns in another dataframe
toyocars1 = toyocars[["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]

# Forming Correlation matrix to check correlation 
corre = toyocars1.corr()
# There is High collinearity between [Age_08_04 and Price]
 #but Price is a output variable
# Plotting scatter plot between all the variables

import seaborn as sns
sns.pairplot(toyocars1)

toyocars1.describe()

# preparing model considering all the variables 

import statsmodels.formula.api as smf
ml1 = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=toyocars1).fit() 

ml1.params
ml1.summary()
#getting r squared values 0.864
#r squared value is greater than 0.80 the standard limit but we are getting pvalues are more than 0.05 for cc and Doors specifically 0.179 and 0.968

ml_Age_08_04=smf.ols("Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data=toyocars1).fit()  
ml_Age_08_04.summary() #0.469
 
ml_KM=smf.ols("KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data = toyocars).fit()  
ml_KM.summary() #0.431

ml_HP=smf.ols("HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight",data = toyocars).fit()  
ml_HP.summary() # 0.295

ml_cc=smf.ols("cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight",data=toyocars1).fit()  
ml_cc.summary() # 0.141

ml_Doors=smf.ols("Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight",data=toyocars1).fit()  
ml_Doors.summary() # 0.135

ml_Gears=smf.ols("Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight",data=toyocars1).fit()  
ml_Gears.summary() # 0.090

ml_Quarterly_Tax=smf.ols("Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight",data=toyocars1).fit()  
ml_Quarterly_Tax.summary() # 0.567

ml_Weight=smf.ols("Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax",data=toyocars1).fit()  
ml_Weight.summary() # 0.603

# now neglecting Weight and building model


ml2 = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax",data=toyocars1).fit() 
ml2.params
ml2.summary()
# Getting R squared value = 0.840 also p value for door i 0 but cc is still at 0.269

# now checking with influence plot on previous model
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

toyocars_new = toyocars.drop(toyocars.index[[80,960,221]],axis=0)

#dropping 80,960,221 as they are influencing most

#building model with removing these values
ml3 = smf.ols("Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight",data = toyocars_new).fit()    
ml3.params
ml3.summary() 

#Finaly R squared is  0.885 with no p values greater than 0.05









#############Start-UP#############









import pandas as pd 
startup50 = pd.read_csv("C:\\Users\\jzsim\\Downloads\\50_Startups.csv")
startup50.head() 
startup50.columns


startup50.rename(columns={'R&D Spend':'Spend','Marketing Spend':'Marketing_Spend'},inplace=True)

# Forming Correlation matrix to check correlation 

startup50.corr()

# There is High collinearity between [R&D Spend and Profit]
 #but Profit is a output variable

# Plotting scatter plot between all the variables

import seaborn as sns
sns.pairplot(startup50)

startup50.describe()

# preparing model considering all the variables 

import statsmodels.formula.api as smf 
ml = smf.ols("Profit~Spend+Administration+Marketing_Spend",data=startup50).fit()
ml.params
ml.summary()

# R squared value = 0.951 but p values are high for Administration = 0.602  and Marketing spend = 0.105 

ml2 = smf.ols("Profit~Spend+Administration",data=startup50).fit()
ml2.summary() 
#rsquared = 0.948 and p high for Administration = 0.289

ml3 = smf.ols("Profit~Administration+Marketing_Spend",data=startup50).fit() 
ml3.summary()
#rsquared = 0.610 and p high for Intercept = 0.259

ml3 = smf.ols("Profit~Spend+Marketing_Spend",data=startup50).fit() 
ml3.summary()
#rsquared = 0.950 

#As we see that Administration is effecting the values 
# Both coefficients p-value became insignificant.

# now checking with influence plot on previous model

import statsmodels.api as sm
sm.graphics.influence_plot(ml)

#dropping 49,48,46 as they are influencing most

startup50_new = startup50.drop(startup50.index[[49,48,46]],axis=0) 

#building model with removing these values
                  
ml4 = smf.ols("Profit~Spend+Administration+Marketing_Spend",data=startup50_new).fit()    
ml4.params
ml4.summary()

# R squared = 0.961  p values high for Administration And Marketing

rsq_Administration = smf.ols("Administration~Spend+Marketing_Spend",data=startup50_new).fit().rsquared  
vif_Administration = 1/(1-rsq_Administration) # 1.23

rsq_Spend = smf.ols("Spend~Administration+Marketing_Spend",data=startup50_new).fit().rsquared  
vif_Spend = 1/(1-rsq_Spend) #2.70

rsq_Marketing_Spend = smf.ols("Marketing_Spend~Spend+Administration",data=startup50_new).fit().rsquared  
vif_Marketing_Spend = 1/(1-rsq_Marketing_Spend)  # 2.68

d = {'Variables':['Admin','Spend','MarketingS'],'VIF': [vif_Administration,vif_Spend,vif_Marketing_Spend]}
vifall = pd.DataFrame(d)
vifall

#as vif for spend is the highest removing it and forming model
ml5 = smf.ols("Profit~Administration+Marketing_Spend",data=startup50_new).fit()    
ml5.summary()
# again model acuracy is dropped and o are high for Intercept

# so selecting model 3 as it is showing the best result









#############Computer#############









import pandas as pd 
compu = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Computer_Data.csv")
compu.head() 
compu.columns


compu.cd[compu.cd == 'no'] = 0
compu.cd[compu.cd == 'yes'] = 1


compu.multi[compu.multi == 'no'] = 0
compu.multi[compu.multi == 'yes'] = 1


compu.premium[compu.premium == 'no'] = 0
compu.premium[compu.premium == 'yes'] = 1


compu.drop(["Unnamed: 0"], axis = 1, inplace = True) 
  
compu.corr()
#no correlation is high

import seaborn as sns
sns.pairplot(compu)

compu.describe()

import statsmodels.formula.api as smf 
ml = smf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=compu).fit()              
ml.params
ml.summary()
# R squared = 0.776

rsq_speed = smf.ols("speed~hd+ram+screen+cd+multi+premium+ads+trend",data=compu).fit().rsquared  
vif_speed = 1/(1-rsq_speed)  # 1.26

rsq_hd = smf.ols("hd~speed+ram+screen+cd+multi+premium+ads+trend",data=compu).fit().rsquared  
vif_hd = 1/(1-rsq_hd) # 4.20

rsq_ram = smf.ols("ram~speed+hd+screen+cd+multi+premium+ads+trend",data=compu).fit().rsquared  
vif_ram = 1/(1-rsq_ram) #2.97

rsq_screen = smf.ols("screen~speed+hd+ram+cd+multi+premium+ads+trend",data=compu).fit().rsquared  
vif_screen = 1/(1-rsq_screen) #1.08 

d = {'Variables':['speed','hd','ram','screen'],'VIF': [vif_speed,vif_hd,vif_ram,vif_screen]}
vifall = pd.DataFrame(d)
vifall

#dropping hd and making model
ml2 = smf.ols("price~speed+ram+screen+cd+multi+premium+ads+trend",data=compu).fit() 
ml2.summary()
# R squared = 0.747

import statsmodels.api as sm
sm.graphics.influence_plot(ml)

compu_new=compu.drop(compu.index[[1440,1700,3783,4477,5960]],axis=0)

ml3 = smf.ols("price~speed+hd+ram+screen+cd+multi+premium+ads+trend",data=compu_new).fit() 
ml3.summary()

# R Squared = 0.777

