import os, pathlib
import numpy as np
from Classifier import Classifier
from DataManager import DataManager

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

f = DataManager(version='1.0')

f.storeData('noise_data/RAVDESS/Actor_01/')

names_cum, X_cum, y_cum = []
names, X, y = f.getDataFromFile('noise_data/RAVDESS/Actor_01/')
names_cum += names
X_cum += X
y_cum += y

c = Classifier('SVM', X_cum, y_cum)

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