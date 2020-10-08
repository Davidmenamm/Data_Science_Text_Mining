# In charge of performing all operations with vector documents

# Imports
from collections import defaultdict
from Document import Document
import math
import numpy as np

# Class VectorSpace
class VectorSpace:
    # Base Document, returns universe of words, for same vector sizes
    def baseVector(self, filt_tweets):
        baseDict = defaultdict(int)
        for _, row in filt_tweets.iterrows():
            for word in row['tweet_text'].split():
                baseDict[word] += 0
        return baseDict
        
    # Generate Documents
    def genDocuments(self, filt_tweets):
        # base Dictionary
        baseDocument = self.baseVector(filt_tweets)
        # Assign Documents
        docs = []
        for _, row in filt_tweets.iterrows():
            # columns
            id = row['tweet_id']
            scrName = row['tweet_screen_name']
            date = row['tweet_date']
            txt = row['tweet_text']
            # create same size vectors
            currentVect = baseDocument
            for word in txt.split():
                currentVect[word] += 1
            # create list of documents
            docs.append(Document(id, scrName, date, currentVect))
        # return
        return docs

    # Constructor
    def __init__(self, filt_tweets):
        self.filt_tweets = filt_tweets
        self.docs = self.genDocuments(filt_tweets)
        self.baseVect = self.baseVector(filt_tweets)
        # to store number of documents currently treating with
        self.numCurrentDocs = 0
    
    # Setter
    def setNumCurrDocs(self, num):
        self.numCurrentDocs = num
    # Getter
    def getDocs(self):
        return self.docs

    # Normalize vector
    def normalize(self, dict_Vector):
        values = np.fromiter(dict_Vector.values(), dtype=float)
        norm = np.linalg.norm(values)
        unitVect = values/norm
        return unitVect

    # Returns a numpy array, for all the keys in the dict
    def vectKeys(self, dict_Vector):
        keys = np.fromiter(dict_Vector.keys(), dtype=float)
        return keys
    
    # Measure the cosine similitude of 2 vectors
    # Receives to numpy arrays, Returns comparison score
    def similitude(self, np_Vect1, np_Vect2):
        score = np.dot(np_Vect1,np_Vect2)
        return score

    # Returns dictionary with number of tweets per candidate
    def tweetsPerCand(self):
        documents = self.docs
        dictCount = defaultdict(int)
        # count tweets per candidate (screenName)
        for doc in documents:
          dictCount[doc.getScreenName()] += 1
        # return
        return dictCount    

    # Calculates average document frequency (df)
    def docf(self, dictTopTerms):
        avgDfDict = defaultdict(int)
        for doc in self.docs:
            avoid_rept = defaultdict(int)
            for word in doc.getVector():
                # to avoid repetead terms for doc freq
                if any(k == word for k in dictTopTerms.keys()):
                    avoid_rept[word] += 1
                # to store only not repeated in doc freq
                for uniqueWord in avoid_rept.keys():
                    avgDfDict[uniqueWord] += 1
        # return
        return avgDfDict
        
    # Receives term frequency (tf,df)
    # Returns all remaing doc analysis information (idf, tf-idf)
    def analysisTab(self, dict_Tf, dict_Docf):
        idf_Dict = dict()
        tf_idf_Dict = dict()
        joinedDict = dict()
        numDocs = self.numCurrentDocs
        # calculating idf
        for word, docFreq in dict_Docf.items():
            # calculations
            idf = math.log(numDocs/docFreq)
            # print('type(idf) ', type(idf))
            # print('idf ', idf)
            # asign value
            idf_Dict[word] = idf

        # calculating tf-idf
        for wordA, termFreq in dict_Tf.items():
            for wordB, idf in idf_Dict.items():
                # print('type(termFreq) ', type(termFreq))
                # print('type(idf) ', type(idf))
                if(wordA == wordB):
                    # calculations
                    tf_idf_Dict[wordA] = termFreq*idf

        # join all dictionaries into one, with all analysis
        idf_Dict = dict(idf_Dict)
        tf_idf_Dict = dict(tf_idf_Dict)
        dicts = [dict_Tf, dict_Docf, idf_Dict, tf_idf_Dict]
        for k in dicts[0]:
            joinedDict[k] = [d[k] for d in dicts]

        # test prints
        # print('dict_Tf', dict_Tf)
        # print('dict_Docf', dict_Docf)
        # print('idf_Dict', idf_Dict)
        # print('tf_idf_Dict', tf_idf_Dict)
        
        # return
        return joinedDict

    # Returns average term frequency for inserted candidates
    # If only one candidate inserted, then its not an average
    def tf(self, dictCandTopUse):
        avgTfDict = defaultdict(int)
        # sum the value of all docts related to a candidate
        for doc in self.docs:
            if any(k == doc.getScreenName() for k,v in dictCandTopUse.items()):
              for word, value in doc.getVector().items():
                  avgTfDict[word] += int(value)
        # integer divide all indices by number of documents:
        # ammount of docs for the docs of the top 10 candidates
        for _, amnt in dictCandTopUse.items():
            self.setNumCurrDocs(self.numCurrentDocs + amnt)
        # print('numDocs', numDocs)
        # make the integer division
        for word, _ in avgTfDict.items():
            # celling division through the negative signs
            avgTfDict[word] = -(avgTfDict[word]//-self.numCurrentDocs)
        # return
        return dict(avgTfDict)

    # Updated version of tf
    def updatedTf(self, dict_Doct):
        avgTfDict = defaultdict(int)
        # sum the value of all docts related to a candidate
        for word, value in dict_Doct.items():
            avgTfDict[word] += int(value)
        # return
        return dict(avgTfDict)
    
    # Receives a Query and returns the corresponding vector
    def queryVector(self, query):
        # create query vector
        currentVect = self.baseVect
        for word in query:
            currentVect[word] += 1
        # normalization, returns numpy array
        qryUnitVect = self.normalize(currentVect)
        #return
        return qryUnitVect
       