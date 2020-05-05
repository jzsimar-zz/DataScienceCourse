########################################1########################################


import pandas as pd
import tweepy 
import re  
import matplotlib.pyplot as plt
from wordcloud import WordCloud

Consumer_key='zujfxDaOwWNyzp2riOgqVKNtP'
Consumer_secret='t5VryUvh6mdQMCKStKGDutQlvMOR5mYH5tgjG019RymTPNpaN3'
Access_key='1248611815-CSKUD8YN11NIWSTYeTBCoBxQ6hN8I2v7fisJAwC'
Access_secret='Bw9Ac9JCqnZOOcyOXdjO0CbcUDqHnW1CEaKkNl0VEqc76'
alltweets = []	
def get_all_tweets(screen_name):
    auth = tweepy.OAuthHandler(Consumer_key,Consumer_secret)
    auth.set_access_token(Access_key,Access_secret)
    api = tweepy.API(auth)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    alltweets.extend(new_tweets)
    
    oldest = alltweets[-1].id - 1
    while len(new_tweets)>0:
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest) #save most recent tweets
        alltweets.extend(new_tweets) #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))
        
    outtweets = [[tweet.created_at,tweet.entities["hashtags"],tweet.entities["user_mentions"],tweet.favorite_count,
                  tweet.geo,tweet.id_str,tweet.lang,tweet.place,tweet.retweet_count,tweet.retweeted,tweet.source,tweet.text,
                  tweet._json["user"]["location"],tweet._json["user"]["name"],tweet._json["user"]["time_zone"],
                  tweet._json["user"]["utc_offset"]] for tweet in alltweets]
    
    
    tweets_df = pd.DataFrame(columns = ["time","hashtags","user_mentions","favorite_count",
                                    "geo","id_str","lang","place","retweet_count","retweeted","source",
                                    "text","location","name","time_zone","utc_offset"])
    tweets_df["time"]  = pd.Series([str(i[0]) for i in outtweets])
    tweets_df["hashtags"] = pd.Series([str(i[1]) for i in outtweets])
    tweets_df["user_mentions"] = pd.Series([str(i[2]) for i in outtweets])
    tweets_df["favorite_count"] = pd.Series([str(i[3]) for i in outtweets])
    tweets_df["geo"] = pd.Series([str(i[4]) for i in outtweets])
    tweets_df["id_str"] = pd.Series([str(i[5]) for i in outtweets])
    tweets_df["lang"] = pd.Series([str(i[6]) for i in outtweets])
    tweets_df["place"] = pd.Series([str(i[7]) for i in outtweets])
    tweets_df["retweet_count"] = pd.Series([str(i[8]) for i in outtweets])
    tweets_df["retweeted"] = pd.Series([str(i[9]) for i in outtweets])
    tweets_df["source"] = pd.Series([str(i[10]) for i in outtweets])
    tweets_df["text"] = pd.Series([str(i[11]) for i in outtweets])
    tweets_df["location"] = pd.Series([str(i[12]) for i in outtweets])
    tweets_df["name"] = pd.Series([str(i[13]) for i in outtweets])
    tweets_df["time_zone"] = pd.Series([str(i[14]) for i in outtweets])
    tweets_df["utc_offset"] = pd.Series([str(i[15]) for i in outtweets])
    tweets_df.to_csv(screen_name+"_tweets.csv")
    return tweets_df

tweet = get_all_tweets("MKBHD")

tweet1=tweet.text

# Joinining all the reviews into single paragraph 
tweet1_string = " ".join(tweet1)



# Removing unwanted symbols incase if exists
tweet1_string = re.sub("[^A-Za-z" "]+"," ",tweet1_string).lower()
tweet1_string = re.sub("[0-9" "]+"," ",tweet1_string)

# words that contained in redmi note 7 reviews
tweet1_words = tweet1_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:\\Users\\Jzsim\\Downloads\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

tweet1_words = [w for w in tweet1_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
tweet1_strings = " ".join(tweet1_words)
#forming wordcloud
wordcloud_MB = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(tweet1_strings)

plt.imshow(wordcloud_MB)

#positive word reviews
with open("C:\\Users\\Jzsim\\Downloads\\positive-words.txt","r") as positive:
  positivewords = positive.read().split("\n")
  
positivew = positivewords[35:]

MBpositive = " ".join ([w for w in tweet1_words if w in positivew])

MBpositivecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(MBpositive)

plt.imshow(MBpositivecloud)

# negative word reviews
with open("C:\\Users\\Jzsim\\Downloads\\negative-words.txt","r") as negative:
  negativewords = negative.read().split("\n")

negativewords = negativewords[35:]

MBnegative = " ".join ([w for w in tweet1_words if w in negativewords])

MBnegativecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(MBnegative)

plt.imshow(MBnegativecloud)



########################################2########################################



import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

redminote7all =[]

for i in range(1,100):
  redminote7=[]  
  url = "https://www.flipkart.com/redmi-note-7-pro-space-black-64-gb/product-reviews/itmfegkx2gufuzhp?pid=MOBFDXZ36Y4DJBGM&lid=LSTMOBFDXZ36Y4DJBGM2SHASI&marketplace=FLIPKART&page="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class",""})
  # Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    redminote7.append(reviews[i].text)  
  redminote7all=redminote7all+redminote7

# Joinining all the reviews into single paragraph 
redminote7all_string = " ".join(redminote7all)



# Removing unwanted symbols incase if exists
redminote7all_string = re.sub("[^A-Za-z" "]+"," ",redminote7all_string).lower()
redminote7all_string = re.sub("[0-9" "]+"," ",redminote7all_string)

# words that contained in redmi note 7 reviews
redminote7all_words = redminote7all_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:\\Users\\Jzsim\\Downloads\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

redminote7all_words = [w for w in redminote7all_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
redminote7all_string = " ".join(redminote7all_words)
#forming wordcloud
wordcloud_n7 = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7all_string)

plt.imshow(wordcloud_n7)
#positive word reviews
with open("C:\\Users\\Jzsim\\Downloads\\positive-words.txt","r") as positive:
  positivewords = positive.read().split("\n")
  
positivew = positivewords[35:]

redminote7positive = " ".join ([w for w in redminote7all_words if w in positivew])

redminote7positivecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7positive)

plt.imshow(redminote7positivecloud)


# negative word reviews
with open("C:\\Users\\Jzsim\\Downloads\\negative-words.txt","r") as negative:
  negativewords = negative.read().split("\n")

negativewords = negativewords[35:]

redminote7negative = " ".join ([w for w in redminote7all_words if w in negativewords])

redminote7negativecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(redminote7negative)

plt.imshow(redminote7negativecloud)


########################################3########################################



import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 
import matplotlib.pyplot as plt
from wordcloud import WordCloud

shawshank =[]
for i in range(1,200):
  imdb=[]  
  url = "https://www.imdb.com/title/tt0111161/reviews?ref_=tt_sa_3"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("div",attrs={"class","text show-more__control"})
  # Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    imdb.append(reviews[i].text)  
  shawshank=shawshank+imdb 
  
  
  
# Joinining all the reviews into single paragraph 
shawshank_string = " ".join(shawshank)



# Removing unwanted symbols incase if exists
shawshank_string = re.sub("[^A-Za-z" "]+"," ",shawshank_string).lower()
shawshank_string = re.sub("[0-9" "]+"," ",shawshank_string)

# words that contained in The Shawshank Redemption reviews
shawshank_words = shawshank_string.split(" ")

#stop_words = stopwords.words('english')

with open("C:\\Users\\Jzsim\\Downloads\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")

shawshank_words = [w for w in shawshank_words if w not in stopwords]

# Joinining all the reviews into single paragraph 
shawshank_string = " ".join(shawshank_words)
#forming wordcloud
wordcloud_TSR = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshank_string)

plt.imshow(wordcloud_TSR)

#positive word reviews
with open("C:\\Users\\Jzsim\\Downloads\\positive-words.txt","r") as positive:
  positivewords = positive.read().split("\n")
  
positivew = positivewords[35:]

shawshankpositive = " ".join ([w for w in shawshank_words if w in positivew])

shawshankpositivecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshankpositive)

plt.imshow(shawshankpositivecloud)


# negative word reviews
with open("C:\\Users\\Jzsim\\Downloads\\negative-words.txt","r") as negative:
  negativewords = negative.read().split("\n")

negativewords = negativewords[35:]

shawshanknegative = " ".join ([w for w in shawshank_words if w in negativewords])

shawshanknegativecloud = WordCloud(
                      background_color='white',
                      width=1800,
                      height=1400
                     ).generate(shawshanknegative)

plt.imshow(shawshanknegativecloud)
