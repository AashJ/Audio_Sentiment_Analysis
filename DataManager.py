import os, pathlib
import numpy as np
from AudioAnalyzer import AudioAnalyzer

class DataManager(object):
    def __init__(self, version='1.0'):
        self.version = version

    def getVector(self, path):
        if self.version == '1.0':
            #TODO: For now, vector is created solely from Audio analysis. With hybrid, we can mix text + audio in vector
            a = AudioAnalyzer('1.0')
            return a.getVector(path)
        else:
            return np.array([0]*100)


    def getDataFromFile(self, filepath):
        file = open(filepath + '/features')
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
        wavfilenames = np.array([file for file in os.listdir(str(path) + '/' + filepath) if file.endswith(".wav")])
        return wavfilenames, np.array([self.getVector(filepath + '/' + str(file)) for file in wavfilenames]), np.array([int(name.split('-')[2]) for name in wavfilenames])

    def storeData(self, filepath):
        #e.g., storeData('noise_data/RAVDESS/Actor_01')
        filenames, X, y = self.getData(filepath + '/sound')

        out = filepath + '/features'
        output = open(out, "w")

        for i in range(len(X)):
            output.write(str(filenames[i]))
            for v in X[i]:
                output.write(' ' + str(v))
            output.write(' ' + str(y[i]))
            output.write('\n')