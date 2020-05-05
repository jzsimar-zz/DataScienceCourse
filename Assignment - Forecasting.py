########################PLASTICSALES########################

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf 

PlasticSales = pd.read_csv("C:\\Users\\jzsim\\Downloads\\PlasticSales.csv")
PlasticSales.Sales.plot()
PlasticSales.describe()
PlasticSales.head(10)

PlasticSales["Date"] = pd.to_datetime(PlasticSales.Month,format="%b-%y")

PlasticSales["Months"] = PlasticSales.Date.dt.strftime("%b")
PlasticSales["Year"] = PlasticSales.Date.dt.strftime("%y") 


sns.boxplot(x="Months",y="Sales",data=PlasticSales)
sns.boxplot(x="Year",y="Sales",data=PlasticSales)

Month_dummies = pd.DataFrame(pd.get_dummies(PlasticSales['Months']))
PlasticSales1 = pd.concat([PlasticSales,Month_dummies],axis = 1)

PlasticSales1["t"] = np.arange(1,61)

PlasticSales1["t_squared"] = PlasticSales1["t"]*PlasticSales1["t"]
PlasticSales1["Log_Sales"] = np.log(PlasticSales1["Sales"])


Train = PlasticSales1.head(48)
Test = PlasticSales1.tail(12)

####################### L I N E A R ##########################

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('Log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('Log_Sales~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('Log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

'''
               MODEL  RMSE_Values
0        rmse_linear   260.937814
1           rmse_Exp   268.693839
2          rmse_Quad   297.406710
3       rmse_add_sea   235.602674
4  rmse_add_sea_quad   218.193876
5      rmse_Mult_sea   239.654321
6  rmse_Mult_add_sea   160.683329

Multiplicative Additive Seasonality BEST FIT

'''
########################AIRLINES########################

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf 

Airlines = pd.read_csv("C:\\Users\\jzsim\\Downloads\\Airlines+Data.csv")
Airlines.Passengers.plot()
Airlines.describe()
Airlines.head()

Airlines["Date"] = pd.to_datetime(Airlines.Month,format="%b-%y")
Airlines["Months"] = Airlines.Date.dt.strftime("%b")
Airlines["Year"] = Airlines.Date.dt.strftime("%Y")


sns.boxplot(x="Months",y="Passengers",data=Airlines)
sns.boxplot(x="Year",y="Passengers",data=Airlines)


Month_Dummies = pd.DataFrame(pd.get_dummies(Airlines['Months']))
Airlines1 = pd.concat([Airlines,Month_Dummies],axis = 1)

Airlines1["t"] = np.arange(1,97)
Airlines1["t_squared"] = Airlines1["t"]*Airlines1["t"]
Airlines1["Log_Passengers"] = np.log(Airlines1["Passengers"])

Train = Airlines1.head(84)
Test = Airlines1.tail(12)

####################### L I N E A R ##########################

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


##################### Exponential ##############################

Exp = smf.ols('Log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('Log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('Log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse

'''
               MODEL  RMSE_Values
0        rmse_linear    53.199237
1           rmse_Exp    46.057361
2          rmse_Quad    48.051889
3       rmse_add_sea   132.819785
4  rmse_add_sea_quad    26.360818
5      rmse_Mult_sea   140.063202
6  rmse_Mult_add_sea    10.519173

Multiplicative Additive Seasonality BEST FIT

'''

########################COCOCOLA########################

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf 

cococola = pd.read_csv("C:\\Users\\jzsim\\Downloads\\CocaCola_Sales_Rawdata.csv")
cococola.Sales.plot()
cococola.describe()
cococola.head()

cococola['Quarters']= 0
cococola['Year'] = 0
for i in range(42):
    p = cococola["Quarter"][i]
    cococola['Quarters'][i]= p[0:2]
    cococola['Year'][i]= p[3:5]
    
    
Quarters_Dummies = pd.DataFrame(pd.get_dummies(cococola['Quarters']))
cococola1 = pd.concat([cococola,Quarters_Dummies],axis = 1)

cococola1["t"] = np.arange(1,43)

cococola1["t_squared"] = cococola1["t"]*cococola1["t"]
cococola1.columns
cococola1["Log_Sales"] = np.log(cococola1["Sales"])

sns.boxplot(x="Quarters",y="Sales",data=cococola1)
sns.boxplot(x="Year",y="Sales",data=cococola1)

Train = cococola1.head(38)
Test = cococola1.tail(4)


####################### L I N E A R ##########################

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('Log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('Log_Sales~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('Log_Sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
'''
               MODEL  RMSE_Values
0        rmse_linear   591.553296
1           rmse_Exp   466.247973
2          rmse_Quad   475.561835
3       rmse_add_sea  1860.023815
4  rmse_add_sea_quad   301.738007
5      rmse_Mult_sea  1963.389640
6  rmse_Mult_add_sea   225.524391

Multiplicative Additive Seasonality BEST FIT

'''