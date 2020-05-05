#####1) Calories_consumed-> predict weight gained using calories consumed

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

calcon=pd.read_csv("C:\\Users\\jzsim\\Downloads\\calories_consumed.csv")

calcon.columns
calcon['Weight gained (grams)']
calcon['Calories Consumed']

#checking for normal data
plt.hist(calcon['Weight gained (grams)'])
plt.boxplot(calcon['Weight gained (grams)'])
plt.hist(calcon['Calories Consumed'])
plt.boxplot(calcon['Calories Consumed'])
calcon.describe()


#as data is not normal going for normalisation
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

calcon_norm = norm_func(calcon.iloc[:,:])
calcon_norm.describe()

plt.hist(calcon_norm['Weight gained (grams)'])
plt.boxplot(calcon_norm['Weight gained (grams)'])
plt.hist(calcon_norm['Calories Consumed'])
plt.boxplot(calcon_norm['Calories Consumed'])


#again going for exponetial data
def sigmoid(x):
    e = np.exp(1)
    y = 1/(1+e**(-x))
    return y

calcon_sigmoid = sigmoid(calcon)
calcon_sigmoid.describe()
plt.hist(calcon_sigmoid['Weight gained (grams)'])
plt.boxplot(calcon_sigmoid['Weight gained (grams)'])
plt.hist(calcon_sigmoid['Calories Consumed'])
plt.boxplot(calcon_sigmoid['Calories Consumed'])

#again going for log function
calcon_log = np.log(calcon)
calcon_log.describe()
plt.hist(calcon_log['Weight gained (grams)'])
plt.boxplot(calcon_log['Weight gained (grams)'])
plt.hist(calcon_log['Calories Consumed'])
plt.boxplot(calcon_log['Calories Consumed'])

#now as the outputs are normal we can go for model building
calcon_log['Weight gained (grams)'].corr(calcon_log['Calories Consumed'])

calcon_log.rename(columns={'Weight gained (grams)':'Weightgain',
                          'Calories Consumed':'Calconsumed'}, 
                  inplace=True)

import statsmodels.formula.api as smf
model=smf.ols("Weightgain~Calconsumed",data=calcon_log).fit()
type(model)
model.params
model.summary() #r squared =  0.846
model.conf_int(0.05)
pred = model.predict(calcon_log)
pred1=np.exp(pred)

import matplotlib.pyplot as plt
plt.scatter(x=calcon_log.Calconsumed,y=calcon_log.Weightgain,color='red');plt.plot(calcon_log.Calconsumed,pred,color='black');plt.xlabel('Calories Consumed');plt.ylabel('Weight Gained')
pred1.corr(calcon_log.Weightgain) # 0.81






#######2) Delivery_time -> Predict delivery time using sorting time








import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


deltime=pd.read_csv("C:\\Users\\jzsim\\Downloads\\delivery_time.csv")
deltime.rename(columns={'Delivery Time':'Delivery_Time','Sorting Time':'Sorting_Time'},inplace=True)

deltime.columns
deltime['Delivery_Time']
deltime['Sorting_Time']

#checking for normal data
plt.hist(deltime.Delivery_Time)
plt.boxplot(deltime.Delivery_Time)
plt.hist(deltime.Sorting_Time)
plt.boxplot(deltime.Sorting_Time)

deltime.describe()


import statsmodels.formula.api as smf
model=smf.ols("Delivery_Time~Sorting_Time",data=deltime).fit()
type(model)
model.params
model.summary() #   r squared value = 0.682
pred = model.predict(deltime) 


import matplotlib.pyplot as plt
plt.scatter(x=deltime.Sorting_Time,y=deltime.Delivery_Time,color='red');plt.plot(deltime.Sorting_Time,pred,color='black');plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')

pred.corr(deltime.Delivery_Time)  # 0.8259972607955325

# as r squared value is low transforming data


# Transforming variables for accuracy
model2=smf.ols("Delivery_Time~np.log(Sorting_Time)",data=deltime).fit()
model2.params
model2.summary() # r squared value =  0.695
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(deltime)
pred2.corr(deltime.Delivery_Time) # 0.8339325279256244
pred2
plt.scatter(x=deltime.Sorting_Time,y=deltime.Delivery_Time,color='red');plt.plot(deltime.Sorting_Time,pred2,color='black');plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')

# Exponential transformation
model3=smf.ols("np.log(Delivery_Time)~Sorting_Time",data=deltime).fit()

model3.params
model3.summary() # r squared value =  0.711
pred_log = model3.predict(deltime)
pred_log
pred3=np.exp(pred_log) 
pred3
pred3.corr(deltime.Delivery_Time)
#0.808578010828926
plt.scatter(x=deltime.Sorting_Time,y=deltime.Delivery_Time,color='red');plt.plot(deltime.Sorting_Time,pred3,color='black');plt.xlabel('Sorting Time');plt.ylabel('Delivery Time')









#####3) Emp_data -> Build a prediction model for Churn_out_rate







import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

empdata=pd.read_csv("C:\\Users\\jzsim\\Downloads\\emp_data.csv")


empdata.columns
empdata.Salary_hike
empdata.Churn_out_rate

#checking for normal data
plt.hist(empdata.Salary_hike)
plt.boxplot(empdata.Salary_hike)
plt.hist(empdata.Churn_out_rate)
plt.boxplot(empdata.Churn_out_rate)

empdata.describe()


import statsmodels.formula.api as smf
model=smf.ols("empdata.Churn_out_rate~empdata.Salary_hike",data=empdata).fit()
type(model)
model.params
model.summary() # r squared value  =  0.831
pred = model.predict(empdata)

import matplotlib.pyplot as plt
plt.scatter(x=empdata.Salary_hike,y=empdata.Churn_out_rate,color='red');plt.plot(empdata.Salary_hike,pred,color='black');plt.xlabel('Salary Hike');plt.ylabel('Churn Out Rate')
pred.corr(empdata.Churn_out_rate) #0.9117216186909112





######4) Salary_hike -> Build a prediction model for Salary_hike






import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

salarydata=pd.read_csv("C:\\Users\\jzsim\\Downloads\\Salary_Data.csv")

salarydata.columns
#checking for normal data
plt.hist(salarydata.YearsExperience)
plt.boxplot(salarydata.YearsExperience)
plt.hist(salarydata.Salary)
plt.boxplot(salarydata.Salary)

salarydata.describe()


import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=salarydata).fit()
type(model)
model.params
model.summary() # r squared value =  0.957
pred = model.predict(salarydata)
import matplotlib.pyplot as plt
plt.scatter(x=salarydata.YearsExperience,y=salarydata.Salary,color='red');plt.plot(salarydata.YearsExperience,pred,color='black');plt.xlabel('Years Experience');plt.ylabel('Salary')

pred.corr(salarydata.Salary) # 0.9782416184887601


