import os, pathlib
import numpy as np
from AudioAnalyzer import AudioAnalyzer

'''
This class manages all the data storage of vector representations, getting data and storing data. We opt to include this
class because we don't want to recompute the same features over and over again - instead, we store the latest features
in a text file and when we need them again, we read from that text file. 

Precondition: The file structure must be formatted as in RAVDESS. That is, for every Actor folder, there must be a 
folder named 'sound', with all the audio. 
'''
class DataManager(object):
    def __init__(self, version='1.0'):
        self.version = version

    '''
    Gets a holistic (taking into account both audio and text) vector representation of a .wav file.
    '''
    def getVector(self, path):
        if self.version == '1.0':
            #TODO: For now, vector is created solely from Audio analysis. With hybrid, we can mix text + audio in vector
            a = AudioAnalyzer('1.0')
            return a.getVector(path)
        else:
            return np.array([0]*100)

    '''
    Extracts data for a particular Actor, using its stored data file. For example, I might call 
    getDataFromFile('noise_data/RAVDESS/Actor_01') to get the data that was stored in Actor_01. This relies on format 
    of data file as specified in storeData().
    '''
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

    '''
    Extracts data for a partiular actor recalculating all the vector representations (i.e., not drawing from the stored
    data). Returns a tuple with file names, vector represetations, and labels.
    '''
    def getData(self, filepath):
        path = pathlib.Path().cwd()
        wavfilenames = np.array([file for file in os.listdir(str(path) + '/' + filepath) if file.endswith(".wav")])
        return wavfilenames, np.array([self.getVector(filepath + '/' + str(file)) for file in wavfilenames]), np.array([int(name.split('-')[2]) for name in wavfilenames])

    '''
    Updates data text file of an Actor. E.g., calling storeData('noise_data/RAVDESS/Actor_01') would update the data
    for all sound associated with Actor_01. The format of the data is the following:
    
    The first element in each line of the data file is the name of the .wav file whose data that line contains. 
    Everything after that first element is a part of the vector representation of that .wav file, except for the last
    element, which is the label (emotion) for that file.
    
    Data is stored in a file called 'features'
    '''
    def storeData(self, filepath):
        filenames, X, y = self.getData(filepath + '/sound')

        out = filepath + '/features'
        output = open(out, "w")

        for i in range(len(X)):
            output.write(str(filenames[i]))
            for v in X[i]:
                output.write(' ' + str(v))
            output.write(' ' + str(y[i]))
            output.write('\n')