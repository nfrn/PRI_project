class Info:
    def __init__(self, label):
        self.manisfests = {}
        self.words = []
        self.label = label

    def addManifest(self,manifestID, manifestTitle, party):
        if manifestID not in self.manisfests.keys():
            self.manisfests[manifestID] = [manifestTitle, party]

    def addWord(self,word):
        if word not in self.words:
            self.words.append(word)

    def __str__(self):
        string = " ENTITY: "
        for word in self.words:
            string += " " + word
        string += "\n"
        string += " LABEL: " + str(self.label)
        string += "\n"
        for key in self.manisfests.keys():
            string += " MANIFEST ID: " + str(key)
            string += " Title: " + str(self.manisfests[key][0])
            string += " Party: " + str(self.manisfests[key][1])
            string += "\n"
        return string
