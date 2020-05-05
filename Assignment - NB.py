#######################SMS#######################
import pandas as pd
import re  
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

sms = pd.read_csv("C:\\Users\\jzsim\\Downloads\\sms_raw_NB.csv",encoding= 'latin-1')

count_vector = CountVectorizer()

smscounts = count_vector.fit_transform(sms['text'])
smsmnb = MultinomialNB()
smsmnb.fit(smscounts, sms['type'])
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
smsmnb.predict(count_vector.transform(["Hello sir we are sendind you this beautiful car for free just log in"]))[0]
smsmnb.predict(count_vector.transform(["we are providing training online from excelr. feel free to join us we give complimentary access to all and jumbo passes limited offer"]))[0]
smsmnb.predict(count_vector.transform(["We are super excited to deliver happines to you. To enjoy seamless shopping experience download our app. We have a welcome gift for you. "]))[0]
smsmnb.predict(count_vector.transform(["Your free ringtone is waiting to be collected. Simply text the password MIX to 85069 to verify. Get Usher and Britney. FML, PO Box 5249, MK17 92H. 450Ppw 1"]))[0]

DXtrain,DXtest,Dytrain,Dytest = train_test_split(sms['text'],sms['type'],test_size=0.3, random_state=0)

sms.type.value_counts()
  
# Joinining all the reviews into single paragraph 
sms_string = " ".join(sms['text'])

# Removing unwanted symbols incase if exists
sms_string = re.sub("[^A-Za-z" "]+"," ",sms_string).lower()
sms_string = re.sub("[0-9" "]+"," ",sms_string)

# words that contained in The Shawshank Redemption reviews
sms_words = sms_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:\\Users\\Jzsim\\Downloads\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

sms_words = [w for w in sms_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
sms_string = " ".join(sms_words)
#forming wordcloud
wordcloud_TSR = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(sms_string)

plt.imshow(wordcloud_TSR)



count_vector.fit(sms_words)
count_vector.get_feature_names()
doc_array = count_vector.transform(sms_words).toarray()
doc_array
frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
frequency_matrix
X_train, X_test, y_train, y_test = train_test_split(sms['text'],sms['type'],test_size=0.20, random_state=0)
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)
accuracy_score(y_test, predictions)



#######################SALARY#######################

import pandas as pd
import re  
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import seaborn as sns

saltrain = pd.read_csv("C:\\Users\\jzsim\\Downloads\\SalaryData_Train.csv")

saltrain.columns
saltrain.describe()
saltrain.plot.hist()

plt.boxplot(saltrain['age'])
plt.boxplot(saltrain['educationno'])
plt.boxplot(saltrain['capitalgain'])
plt.boxplot(saltrain['hoursperweek'])
sns.boxplot(x="Salary",y="age",data=saltrain,palette = "hls")
sns.boxplot(x="Salary",y="educationno",data=saltrain,palette = "hls")
sns.boxplot(x="Salary",y="capitalgain",data=saltrain,palette = "hls")
sns.boxplot(x="Salary",y="hoursperweek",data=saltrain,palette = "hls")

saltrain.isnull().sum()

plt.figure(figsize=(20,10))
c=saltrain.corr()
sns.heatmap(c,cmap='BrBG',annot=True)
c
saltrain.info()
categorical = [var for var in saltrain.columns if saltrain[var].dtype=='O']
saltrain[categorical].isnull().sum()

X = saltrain.drop(['Salary'], axis=1)
y = saltrain['Salary']

import category_encoders as ce
# encode remaining variables with one-hot encoding
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)
X_train.head(10)
X_test = encoder.transform(X_test)
X_test.head()

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

(5444+1808)/(5444+1354+443+1808)

