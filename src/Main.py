# Program that ...

# Imports
from Coordinator import Coordinator
import matplotlib.pyplot as plt

# print dictionary function
def toStringDict(dict):
  strDict = ''
  for k,v in dict.items():
    strDict += f'{k} -> {v}\n'
  return strDict


# Paths
# candListPath = r'src\data\candidates.csv'
candListPath = r'src\data\candidates.csv'
tweetsPath = r'src\data\tweets.pkl'
outputTxtPath = r'src\data\output.txt'

# Initiate program:
coord = Coordinator(candListPath, tweetsPath)

# Read the input data, and store in corresponding places
coord.readData()

# a) Get top 10 candidates that used twitter the most
print('a) Get top 10 candidates, that used twitter the most:')
dictTopUse = coord.topCandTwtUse()
# print
strTopUse = toStringDict(dictTopUse)
print (strTopUse)

# b) Get top 10 words for top 10 candidates use
print('b) Get top 10 words, used by candidates that use twitter the most:')
topWords = coord.topWordsTwtUse()
# print
strTopWords = toStringDict(topWords)
print (strTopWords)

# c) Generate timeline of top ten words per candidate
print('c) Generate timeline of top ten words per candidate:')
selectdCand, dateDict = coord.topTimeline(2)
# print
print(f'Candidate is: {selectdCand}')
# strTimeLine = toStringDict(topWords)
# for k,vals in dateDict:
#     print(k)
#     print (vals)
#     for val in vals:
#         str_V = ''
#         for v in val:
#           str_V += f' {v}'
#         print(str_V)
print(dateDict)

# d) Ranking for each candidate, in base of user introduced query
print('d) Ranking for candidate, in base of user introduced query:')
query = input('Enter the query:')
ranking = coord.candRanking(query)
# print
strRanking = toStringDict(ranking)
print (strRanking)

# e) Which candidates from literal a), had a similar campaign (tweet similarities)
print('e) Which candidates, had a similar campaign:')
pd_simCampaign = coord.simCampaign()
print(pd_simCampaign)

# For testing, print to file
# f = open(outputTxtPath, 'w')
# f.write(str(filt_tweets.head(50)))
# f.close()

