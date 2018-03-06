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

def getFeatures(path):
    # e.g., features = getFeatures('noise_data/user/5.wav')
    # ideas for which features to extract: http://www.fon.hum.uva.nl/praat/
    signal, samplingRate = lbr.load(path)

    # Compute MFCC features from the raw signal
    frame_ms = 30
    mfcc = lbr.feature.mfcc(y=signal, sr=samplingRate, hop_length=int(samplingRate*frame_ms/1000), n_mfcc=13)

    # And the first-order differences (delta features)
    mfcc_delta = lbr.feature.delta(mfcc)

    return (mfcc, mfcc_delta)

def getVector(path):
    mfcc, mfcc_delta = getFeatures(path)
    return mfcc_delta[0]

def getMatrix(filepath):
    path = pathlib.Path().cwd()
    wavfilenames = [file for file in os.listdir(str(path) + '/' + str(filepath) + 'sound') if file.endswith(".wav")]
    return wavfilenames, [getVector(filepath + 'sound/' + str(file)) for file in wavfilenames], [int(name.split('-')[2]) for name in wavfilenames]

def storeData(filepath):
    #e.g., storeData('noise_data/RAVDESS/Actor_01/')
    filenames, X, y = getMatrix(filepath)

    out = filepath + 'features'
    output = open(out, "w")

    for i in range(len(X)):
        output.write(str(filenames[i]))
        for v in X[i]:
            output.write(' ' + str(v))
        output.write('\n')

def predict(filepath, label):
    _, matrix = getMatrix(filepath)

_, X, y = getMatrix('noise_data/RAVDESS/Actor_01/')
print(str(y))
c = Classifier('SVM', X, y)
c.predictSVM()

#http://samcarcagno.altervista.org/blog/basic-sound-processing-python/


