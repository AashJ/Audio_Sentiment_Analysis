import os, pathlib
import numpy as np
from AudioAnalyzer import AudioAnalyzer
from TextAnalyzer import TextAnalyzer
import os
import wave
import pylab
from PIL import Image

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
    
    Postcondition: returns a list; not a numpy array
    '''
    def getVector(self, path):
        if self.version == '1.0':
            #TODO: come up with way to combine both vector representations
            a = AudioAnalyzer(self.version)
            #For now, vector is created using only the audio vector
            #t = TextAnalyzer(self.version)
            vector = a.getVector(path)
            return vector
        else:
            return [0]*100

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

        size = len(wavfilenames)
        c = 0
        p = 0
        X = []
        print('----' + '0% stored...', end="\r")
        for file in wavfilenames:
            X += [self.getVector(filepath + '/' + str(file))]
            DataManager.wave_to_spectogram(file)
            if round(100*c/size, 1) != p:
                p = round(100*c/size, 1)
                print('----' + str(p) + '% stored...', end="\r")
            c += 1
        print('----' + '100% stored.')
        return wavfilenames, np.array(X), np.array([int(name.split('-')[2]) for name in wavfilenames])

    '''
    Updates data text file of an Actor. E.g., calling storeData('noise_data/RAVDESS/Actor_01') would update the data
    for all sound associated with Actor_01. The format of the data is the following:
    
    The first element in each line of the data file is the name of the .wav file whose data that line contains. 
    Everything after that first element is a part of the vector representation of that .wav file, except for the last
    element, which is the label (emotion) for that file.
    
    Data is stored in a file called 'features'. Note that storing data takes a long time, but after this is done it is
    quick to simply read the data from the file in which it was stored. Hence, use this method sparingly.
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

    @staticmethod
    def spectogram_dir(dir):
        if not os.path.exists(dir + 'spectogram'):
            os.makedirs(dir + 'spectogram')

        for file in os.listdir(dir + 'sound'):
            # for some reason, 112--6-.wav doesn't work.
            if file != '.DS_Store' and file != '112--6-.wav' and file != '162--5-.wav':
                DataManager.graph_spectrogram(dir + 'sound/' + file)


    @staticmethod
    def graph_spectrogram(wav_file):
        print(wav_file)
        sound_info, frame_rate = DataManager.get_wav_info(wav_file)
        pylab.figure(num=None, figsize=(19, 12))
        pylab.subplot(111)
        pylab.title('spectrogram of %r' % wav_file)
        pylab.specgram(sound_info, Fs=frame_rate)
        new_file = wav_file.replace('sound', 'spectogram')
        pylab.savefig(new_file + '.png')
        pylab.close()
        DataManager.crop_and_resize(new_file + '.png')

    @staticmethod
    def crop_and_resize(img):
        image = Image.open(img)
        tgt = image.crop((235, 144, 1711, 1067))
        tgt.thumbnail((256, 256), Image.ANTIALIAS)
        tgt.save(img)

    @staticmethod
    def get_wav_info(wav_file):
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        sound_info = pylab.fromstring(frames, 'int16')
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate