import numpy as np
from textblob import TextBlob
from Microphone import Recorder

'''
Deal with everything to do with text here. This takes as input a filepath to a .wav file, gets the text that was said
in that .wav file and returns data about that text.'version' specifies which version feature representation the user 
wants to use.

We use textblob for feature extraction:
https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis
'''
class TextAnalyzer:
    def __init__(self, version='1.0'):
        self.version = version

    def getFeatures(self, path):
        #TODO
        text = Recorder.transcribe(path)
        blob = TextBlob(text)
        polarity, subjectivity = blob.sentiment
        return (polarity, subjectivity)

    def getVector(self, path):
        #TODO
        #ideas: each coordinate is a prediction from imported classifier?
        data = self.getFeatures(path)
        vector = []
        for arg in data:
            vector += [arg]

        return np.array(vector)

#sanity check
'''
text = Recorder.transcribe('noise_data/RAVDESS/Actor_01/sound/03-01-01-01-01-01-01.wav')
print(text)
blob = TextBlob(text)
polarity, subjectivity = blob.sentiment
print(str(polarity) + "||" + str(subjectivity))

text = "I hate you, asshole"
print(text)
blob = TextBlob(text)
polarity, subjectivity = blob.sentiment
print(str(polarity) + "||" + str(subjectivity))

text = "I love flowers!"
print(text)
blob = TextBlob(text)
polarity, subjectivity = blob.sentiment
print(str(polarity) + "||" + str(subjectivity))
'''
