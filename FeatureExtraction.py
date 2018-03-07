import os, pathlib
import librosa as lbr
#http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
from sklearn import svm
import numpy as np

class Classifier(object):
    def __init__(self, algorithm, x_train, y_train, iterations=1, averaged=False, eta=1.5, alpha=1.1):
        self.alg = algorithm
        # create a dictionary of classifiers for each label
        self.classifiers = {}
        self.possibleLabels = list(set(y_train))

        if algorithm == 'SVM':
            for label in self.possibleLabels:
                clf = svm.LinearSVC()
                y_split = np.array([int(y == label) for y in y_train])
                clf.fit(x_train, y_split)
                self.classifiers[label] = clf

    def predict(self, x):
        if self.alg == 'SVM':
            for label in self.possibleLabels:
                prediction = self.classifiers[label].predict([x])[0]
                if prediction == 1:
                    return prediction
            return 'This example didn\'t fall into any of the categories'

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

        #TODO figure out what more features to extract

        return (mfcc, mfcc_delta)

    def getVector(self, path):
        if self.version == '1.0':
            mfcc, mfcc_delta = self.getFeatures(path)

            #TODO create vector representations of .wav files

            vector = [0]*1000

            count = 0
            for list in mfcc_delta:
                for i in range(len(list)):
                    if count < 1000:
                        vector[count] = list[i]
                    count += 1
                if count > 500:
                    break

            for list in mfcc:
                for i in range(len(list)):
                    if count < 1000:
                        vector[count] = list[i]
                    count += 1
                if count > 999:
                    break

            return vector
        else:
            return np.array([0]*100)

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
            output.write(' ' + str(y[i]))
            output.write('\n')

def getEmotion(label):
    if label == 1:
        return 'neutral'

    if label == 2:
        return 'calm'

    if label == 3:
        return 'happy'

    if label == 4:
        return 'sad'

    if label == 5:
        return 'angry'

    if label == 6:
        return 'fearful'

    if label == 7:
        return 'disgust'

    if label == 8:
        return 'surprised'

f = FeatureSet(version='1.0')
f.storeData('noise_data/RAVDESS/Actor_01/')

names, X, y = f.getDataFromFile('noise_data/RAVDESS/Actor_01/')
c = Classifier('SVM', X, y)

'''
for i in range(1, 11):
    label = c.predict(f.getVector('noise_data/user/' + str(i) + '.wav'))
    print(str(i) + '.wav')
    printEmotion(label)
'''

#tests
path = pathlib.Path().cwd()
wavfilenames = np.array([file for file in os.listdir(str(path) + '/noise_data/RAVDESS/Actor_01/sound') if file.endswith(".wav")])
emotions = [getEmotion(int(name.split('-')[2])) for name in wavfilenames]
for i in range(len(wavfilenames)):
    file = wavfilenames[i]
    emotion = emotions[i]
    label = c.predict(f.getVector('noise_data/RAVDESS/Actor_01/sound/' + str(file)))
    print('noise_data/RAVDESS/Actor_01/sound/' + str(file) + ': ' + emotion + ' | ' + getEmotion(label))

#http://samcarcagno.altervista.org/blog/basic-sound-processing-python/
