# Represents the documents used in the text mining process

# Class Document
class Document:
    # Constructor
    def __init__(self, id, screenName, date, vector):
        self.id = id
        self.screenName = screenName       
        self.date = date
        self.vector = vector
    # getter
    def getId(self):
        return self.id
    def getScreenName(self):
        return self.screenName
    def getDate(self):
        return self.date
    def getVector(self):
        return self.vector