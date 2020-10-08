# Read.py :
# Reads candidates list (.csv)
# Reads tweets (.pkl)
# Reads candidates manifestos (.pdf)
# Returns candidates list data, as ...
# Returns candidates tweets as documents, per tweet, and per candidate
# Returns manifestos as documents also

# Imports
import numpy as np
import pandas as pd
import pickle


# Class Read
class Read:
    # constructor
    def __init__(self, candListPath, tweetsPath):
        self.tweetsPath = tweetsPath
        self.candListPath = candListPath

    # funtion to read candidate list (.csv)
    def readCandList(self):
        path = self.candListPath
        cd_list = pd.read_csv(path, delimiter = ',')
        return cd_list

    # function to read tweets (.pkl)
    def readTweets(self):
        path = self.tweetsPath

        # Read tweets pd_dataframe,  as pickle obj
        pickle_in = open(path, 'rb')
        pd_dataFrame = pickle.load(pickle_in)
        pickle_in.close()
        return pd_dataFrame

    # filter tweets, only does from candidates remain
    def filterTweets(self, pd_candList, pd_tweets):
        pd_filter = pd.DataFrame()
        for _, row in pd_candList.iterrows():
            # get twitter_id of candidate
            twitter_id = row['twitter_screen_name']
            # get corresponding tweets from that candidate
            corr_tweets = pd_tweets.loc[pd_tweets['tweet_screen_name'] == twitter_id ]
            # append data frames
            pd_filter = pd_filter.append(corr_tweets)
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        return pd_filter
            