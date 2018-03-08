import os, pathlib
import librosa as lbr
#http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
import numpy as np

class DataManager(object):
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