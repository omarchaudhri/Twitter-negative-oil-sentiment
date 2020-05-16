import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime
import time


#import data into dataframe
data = pd.read_csv("data_oil_price.csv")
data.dropna()


#getting only english tweets as textblob can not work on other languages and so result would be not useful
data = data[data['lang'] == "en"]


#dropping unnecessary columns
data.drop(["source","id_str","truncated","full_tweet","user_location","user_description","user_time_zone"], axis = 1, inplace=True)

#Setting the date to a format that can be set in ascending order
split_values = data['created_at'].str.split(" ",expand = True)
split_values.columns = ["Day of week","Month","Day","Time of tweet","nanosecond","year"]
split_values.drop(["nanosecond","year","Day of week"],axis = 1,inplace = True)
split_values["Time of tweet"] = split_values["Time of tweet"].str.replace(':','')
split_values["Day"] = split_values["Day"].astype(int)
split_values["Time of tweet"] = split_values["Time of tweet"].astype(int)

combined_df = split_values.join(data)


#ordering by day and then time of day
combined_df.sort_values(["Day","Time of tweet"],ascending=True,inplace=True)


#There were too many duplicates in the dataset, a burst of spam tweets from a user with the same timestamp
#I filtered them using the id field and only kept the first instance
#combined_df.drop_duplicates(subset = "text", keep = "first", inplace = True)
combined_df.drop_duplicates(subset = "id", keep = "first", inplace = True)

#selecting data for 20th and 21st only
combined_df = combined_df[combined_df["Month"] == "Apr"]
combined_df = combined_df[combined_df["Day"] >= 20 ]

#Tweets text contain URLs which are removed
no_https = combined_df['text'].str.split("https", expand = True)
drop_cols = [1,2,3,4,5]
no_https.drop(no_https.columns[drop_cols],axis = 1,inplace = True)
no_https.columns = ["Tweet text"]
clean_data = no_https.join(combined_df)
clean_data.drop('text',axis = 1, inplace = True)


#it was 420 that day, a festival for some people, removing multiple spam 420 tweets keeping only at max  to improve
#accuracy of cleaning
#clean_data.drop_duplicates(subset = "Tweet text", keep = "first", inplace = True)


#adding columns for polarity and subjectivty scores using textblob
clean_data['polarity'] = clean_data['Tweet text'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
clean_data['subjectivity'] = clean_data['Tweet text'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)

#saving cleaned data as csv
clean_data.to_csv(r'C:\Users\asap1\.PyCharmCE2019.2\config\scratches\English_tweets_cleaned.csv', index = False)

entire_data_mean_sentiment = []
entire_data_mean_sentiment.append(clean_data['polarity'].mean())
entire_data_mean_sentiment.append(clean_data['subjectivity'].mean())


#counting individual tweets by a user
count_series = combined_df['user_id'].value_counts()
count_df = count_series.to_frame().reset_index()
count_df.columns = ['user_id','freq']
count_df.to_csv(r'C:\Users\asap1\.PyCharmCE2019.2\config\scratches\count_df.csv', index = False)



#Finding out the twitter addicts who are tweeting more than 10 times during two days


twitter_addict = count_df['freq'] > 10
twitter_addict = count_df[twitter_addict]

df = pd.merge(clean_data, twitter_addict, on=['user_id'], how='left', indicator='Exist')
df['true_user'] = np.where(df.Exist == 'both', True, False)

criteria = df['true_user'] == True
twitter_addict_tweets = df[criteria]
print(twitter_addict_tweets['polarity'])

int_tweets = twitter_addict_tweets['polarity'] > -1
int_tweets = twitter_addict_tweets[int_tweets]
int_tweets.drop_duplicates(subset = "Time of tweet", keep = "first", inplace = True)
int_tweets.to_csv(r'C:\Users\asap1\.PyCharmCE2019.2\config\scratches\top tweeters.csv', index = False)
