#####################Startup#####################

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
import matplotlib.pyplot as plt

startup0 = pd.read_csv("C:\\Users\\jzsim\\Downloads\\50_Startups.csv")
startup0.columns
startup0.describe()
startup0.plot.hist()
startup0.plot.bar()
startup0.plot.box()

plt.boxplot(startup0['R&D Spend'])
plt.boxplot(startup0['Administration'])
plt.boxplot(startup0['Marketing Spend'])
plt.boxplot(startup0['Profit'])

startup0.isnull().sum()

startup1 = pd.get_dummies(startup0)
startup1.columns

column_names = ['R&D Spend', 'Administration', 'Marketing Spend','State_California', 'State_Florida', 'State_New York', 'Profit']
startup = startup1.reindex(columns=column_names)

cont_model = Sequential()
cont_model.add(Dense(500,input_dim=6,activation="relu"))
cont_model.add(Dense(4000,activation="relu"))
cont_model.add(Dense(2000,activation="relu"))
cont_model.add(Dense(1000,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])


column_names = list(startup.columns)
predictors = column_names[0:6]
predictors
target = column_names[6]
target
first_model = cont_model
first_model.fit(np.array(startup[predictors]),np.array(startup[target]),epochs=30)
pred_train = first_model.predict(np.array(startup[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-startup[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,startup[target],"bo")
np.corrcoef(pred_train,startup[target]) 

from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title="My first neural network")






#####################Forest Fire#####################

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
import seaborn as sns

forestfire = pd.read_csv("C:\\Users\\jzsim\\Downloads\\forestfires.csv")

forestfire.head()
forestfire.describe()
forestfire.columns

forestfire.plot.box()
forestfire.plot.hist()
forestfire.plot.bar()

forestfire.isnull().sum()

sns.boxplot(x="size_category",y="FFMC",data=forestfire,palette = "hls")
sns.boxplot(x="size_category",y="DMC",data=forestfire,palette = "hls")
sns.boxplot(x="size_category",y="DC",data=forestfire,palette = "hls")
sns.boxplot(x="size_category",y="ISI",data=forestfire,palette = "hls")
sns.boxplot(x="size_category",y="temp",data=forestfire,palette = "hls")
sns.boxplot(x="FFMC",y="size_category",data=forestfire,palette = "hls")
sns.boxplot(x="DMC",y="size_category",data=forestfire,palette = "hls")


plt.scatter(x=forestfire['FFMC'],y=forestfire['DMC'],color='red')

forestfire.size_category[forestfire.size_category == 'small'] = 0
forestfire.size_category[forestfire.size_category =='large'] = 0

column_names = list(forestfire.columns)
predictors = column_names[2:30]
predictors
target = column_names[30]
target


cont_model = Sequential()
cont_model.add(Dense(500,input_dim=28,activation="relu"))
cont_model.add(Dense(4000,activation="relu"))
cont_model.add(Dense(2000,activation="relu"))
cont_model.add(Dense(1000,activation="relu"))
cont_model.add(Dense(1,kernel_initializer="normal"))
cont_model.compile(loss="mean_squared_error",optimizer = "adam",metrics = ["mse"])

first_model = cont_model
first_model.fit(np.array(forestfire[predictors]),np.array(forestfire[target]),epochs=5)
pred_train = first_model.predict(np.array(forestfire[predictors]))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-forestfire[target])**2))

import matplotlib.pyplot as plt
plt.plot(pred_train,forestfire[target],"bo")

np.corrcoef(pred_train,forestfire[target].astype(np.float64)) 

from ann_visualizer.visualize import ann_viz
ann_viz(first_model, title="My first neural network")




