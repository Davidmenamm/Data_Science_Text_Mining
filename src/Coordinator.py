# Coordinates all the text mining process, solving all literals

# Imports
from Read import Read
from VectorSpace import VectorSpace
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# Class Coordinator
class Coordinator:
    # Constructor
    def __init__(self, candListPath, tweetsPath):
        # paths
        self.candListPath = candListPath
        self.tweetsPath = tweetsPath   
        # list of candidates    
        self.candList = None
        # operations with tweets, and creation of documents
        self.vectorSpace = None
        # principal methods outputs
        self.candTopUse = None
        self.allAnalysDict = None
        self.tfTimeLine = []
        self.tf = None
        self.sampleSize = 16000 # for testing, to run program faster
    
    # to read input data & save in the corresponding forms
    def readData(self):
        candListPath = self.candListPath
        tweetsPath = self.tweetsPath
        # initiate reading
        read = Read(candListPath, tweetsPath)
        # read candidate list and save
        self.candList = read.readCandList()
        # read, filter tweets and save
        tweets = read.readTweets().sample(self.sampleSize) # could change with sample
        filt_tweets = read.filterTweets(self.candList,tweets)
        # Create new document vector space
        # and generate respective documents
        self.vectorSpace = VectorSpace(filt_tweets)

    # a) top 10 candidates that used twitter the most, during their campaign
    def topCandTwtUse(self):
      # get dictionary with num of tweets per candidate
      self.candTopUse = self.vectorSpace.tweetsPerCand()
      # sort higher values on top (descending order)
      self.candTopUse = sorted(self.candTopUse.items(), key=lambda kv: kv[1], reverse = True)
      self.candTopUse = dict(self.candTopUse[:10]) # Top 10 candidates
      # return
      return self.candTopUse

    # b) top 10 words used by candidates that use twitter the most
    # analyzed in terms of (tf, df, idf, tf-idf)
    def topWordsTwtUse(self):
        # term frequency (tf)
        self.tf = self.vectorSpace.tf(self.candTopUse)
        # sort higher values on top (descending order)
        self.tf = sorted(self.tf.items(), key=lambda kv: kv[1], reverse=True)
        self.tf = dict(self.tf[:10]) # Top 10 words or terms

        # documentent frequency (df)
        avgDocf = self.vectorSpace.docf(self.tf)

        # Retrieve all remaining analysis
        self.allAnalysDict = self.vectorSpace.analysisTab(self.tf, avgDocf)

        # return
        return self.allAnalysDict
    
    # topWordsTwtUse update
    # def updTopWords(self):
    #     # analyze docs of corresponding candidates
    #     analyzeDocs = defaultdict(list)
    #     for doc in self.vectorSpace.getDocs():
    #         scrName = doc.getScreenName()
    #         if any(k == scrName for k,v in self.candTopUse.items()):
    #             vect_Dict = doc.getVector()
    #             # calculate tf
    #             tf_dict = self.vectorSpace.updatedTf(vect_Dict)
    #             analyzeDocs[scrName].append()


    # c) Get top 10 words per candidate, for timeline
    def topTimeline(self, cand_Idx):
        # get dates per top candidate, and word frequency per date
        dateWords = defaultdict(list)
        selectedCand = None
        # iterate through documents
        for doc in self.vectorSpace.getDocs():
            # doc methods
            scrName = doc.getScreenName()
            date = doc.getDate()
            wordFreq = doc.getVector()
            # filter only top candidates
            selectedCand = list((self.candTopUse.keys()))[cand_Idx]
            if any(k == scrName for k in selectedCand):
                dateWords[date].append(wordFreq)
        # join word freq, for each date
        sumDict = {}
        dateWords_Joined = {}
        for date, ls_dict_wf in dateWords.items():
            # sum frequency of dicts of wf, associated to each date
            for dict_wf in ls_dict_wf:
                sumDict = dict(Counter(sumDict) + dict(Counter(dict_wf)))
            dateWords[date] = [].append(sumDict)
            sumDict = {}
        
        # build respective axis vectors to graph later:
        finalDict = defaultdict(list)
        for topWord, _ in self.tf.items(): # go through top words
            dates = []
            freq = []
            for date, wordFreq in dateWords.items():
                # filter that it contains top word
                if any(k == topWord for k in dateWords.keys()):
                    dates.append(date)
                    freq.append((dateWords[date])[0])
            # add all double lists, to final dictionary
            finalDict[topWord].append(dates)
            finalDict[topWord].append(freq)
        # return
        return selectedCand, finalDict

        # # get words per candidate
        # for cand,num in self.candTopUse.items():
        #     candDict = {
        #         cand:num
        #     }
        #     current = self.vectorSpace.tf(candDict)
        #     current = sorted(current.items(), key=lambda kv: kv[1], reverse=True)
        #     current = dict(current[:10])
        #     self.tfTimeLine.append(current)
        # # return
        # return 'self.tfTimeLine'
    
    # d) Ranking for each candidate, based on user query
    def candRanking(self, query):
        # generate query vector
        queryLs = query.split()
        np_QueryVect = self.vectorSpace.queryVector(queryLs)
        # get dictionary with candidates associated vectors
        candVectors = defaultdict(list)
        for doc in self.vectorSpace.getDocs():
            # filter only candidates, not other users
            if any(k == doc.getScreenName() for k,v in self.candTopUse.items()):
                # save candidate associated vectors
                currentDoc = doc.getVector()
                vect = self.vectorSpace.normalize(currentDoc)
                candVectors[doc.getScreenName()].append(vect)
        # calculate de score for each candidate
        candScore = defaultdict(int)
        for screenName, vectors in candVectors.items():
            for vect in vectors:
                similitude = np.dot(np_QueryVect, vect)
                candScore[screenName] += similitude
        # sort dictionary in descending order
        candScore = sorted(candScore.items(), key=lambda kv: kv[1], reverse=True)
        # return
        return dict(candScore)

    # e) Returns pd_Dataframe of top twitter use candidates, campaign similarities
    def simCampaign(self):
        # get dictionary with candidates associated vectors
        candVectors = defaultdict(list)
        for doc in self.vectorSpace.getDocs():
            # filter only candidates, not other users
            if any(k == doc.getScreenName() for k,v in self.candTopUse.items()):
                # save candidate associated vectors
                currentDoc = doc.getVector()
                vect = self.vectorSpace.normalize(currentDoc)
                candVectors[doc.getScreenName()].append(vect)
        # calculate de score for each candidate
        candScore = defaultdict(list)
        # for every candidate, dot-product its vectors, with all other candidates vectors
        for screenNameA, vectorsA in candVectors.items():
            for screenNameB, vectorsB in candVectors.items():
                similitude = 0
                for vectA in vectorsA:
                    # for calculating one of the similitude scores
                    for vectB in vectorsB:
                        similitude += np.dot(vectA, vectB)
                # store the similitude score
                candScore[screenNameA].append(similitude)

        # create pd_dataframe to store all info orderly
        pd_CandScore = pd.DataFrame.from_dict(candScore,orient='index')
        pd_CandScore = pd_CandScore.transpose()
        # return
        return pd_CandScore