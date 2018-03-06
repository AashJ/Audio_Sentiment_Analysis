import os, pathlib
import librosa as lbr
#http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
from sklearn import svm
import numpy as np

class Classifier(object):

    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged=False, eta=1.5, alpha=1.1):
        self.alg = algorithm
        if algorithm == 'SVM':
            x_train = self.v.fit_transform(x_train)
            y_train = np.array(y_train)
            self.clf = svm.LinearSVC()
            self.clf.fit(x_train, y_train)

    def predict(self, x, y):
        if self.alg == 'SVM':
            return self.clf.predict(x)[0]

class FeatureSet(object):
    def __init__(self, version='1.0'):
        self.version = version

    def getFeatures(self, path):
        # e.g., features = getFeatures('noise_data/user/5.wav')
        # ideas for which features to extract: http://www.fon.hum.uva.nl/praat/
        signal, samplingRate = lbr.load(path)

        # Compute MFCC features from the raw signal
        frame_ms = 30
        mfcc = lbr.feature.mfcc(y=signal, sr=samplingRate, hop_length=int(samplingRate*frame_ms/1000), n_mfcc=13)

        # And the first-order differences (delta features)
        mfcc_delta = lbr.feature.delta(mfcc)

        return (mfcc, mfcc_delta)

    def getVector(self, path):
        if self.version == '1.0':
            mfcc, mfcc_delta = self.getFeatures(path)




            #create vector representations




            return mfcc_delta[0]
        else:
            return [0]*100

    def getDataFromFile(self, filepath):
        file = open(filepath + 'features')
        # creates a list of the lines in the source data
        lines = file.readlines()
        split_lines = []
        for line in lines:
            data = line.split()
            split_lines += [(data[0], np.array(data[1:-1]).astype(np.float), int(data[-1]))]

        filenames, vectors, labels = zip(*split_lines)
        return np.array(filenames), np.array(vectors), np.array(labels)

    def getData(self, filepath):
        path = pathlib.Path().cwd()
        wavfilenames = np.array([file for file in os.listdir(str(path) + '/' + str(filepath) + 'sound') if file.endswith(".wav")])
        return wavfilenames, np.array([self.getVector(filepath + 'sound/' + str(file)) for file in wavfilenames]), np.array([int(name.split('-')[2]) for name in wavfilenames])

    def storeData(self, filepath):
        #e.g., storeData('noise_data/RAVDESS/Actor_01/')
        filenames, X, y = self.getData(filepath)

        out = filepath + 'features'
        output = open(out, "w")

        for i in range(len(X)):
            output.write(str(filenames[i]))
            for v in X[i]:
                output.write(' ' + str(v))
            output.write('\n')


f = FeatureSet('1.1')
f.storeData('noise_data/RAVDESS/Actor_01/')
#_, X, y = f.getData('noise_data/RAVDESS/Actor_01/')
#print(str(y))
#c = Classifier('SVM', X, y)
#c.predictSVM()

#http://samcarcagno.altervista.org/blog/basic-sound-processing-python/

