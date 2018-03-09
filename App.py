from Classifier import Classifier
from DataManager import DataManager
from Microphone import Recorder
from sklearn.metrics import confusion_matrix
import numpy as np
import pathlib, os
import warnings
warnings.filterwarnings("ignore")

#Gets the emotion that each label corresponds to. See 'about' in RAVDESS for more info
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

    return 'N/A'

def getTrainingData(pathList, f):
    # Extract and compile data from all specified directories
    names_cum, X_cum, y_cum = [], [], []
    for path in pathList:
        names, X, y = f.getDataFromFile(path)
        names, X, y = names.tolist(), X.tolist(), y.tolist()
        names_cum += names
        X_cum += X
        y_cum += y
    return np.array(names_cum), np.array(X_cum), np.array(y_cum)

def storeData(pathList, f):
    # Store data (if not stored already from previous run)
    for path in pathList:
        print('--Storing ' + str(path) + ' ...')
        f.storeData(path)

'''
#file formatting junk code (don't delete)
i = 0
init_path = pathlib.Path().cwd()
fullpath = str(init_path) + '/noise_data/RAVDESS/temp'
for file in os.listdir(fullpath):
    i += 1
    if file.endswith(".wav"):
        os.rename(fullpath + '/' + str(file), fullpath + '/' + str(i) + '--8-.wav')
'''

'''
#record a snippet
print('\n--------------RECORDING--------------')
r = Recorder()
r.recordWAV(8)
'''

print('\n---------------RUNNING---------------')
f = DataManager(version='1.0')
train_pathList = ['noise_data/RAVDESS/Actor_02', 'noise_data/RAVDESS/Actor_03']

print('Storing data...')
#storeData(train_pathList, f)

print('Loading data...')
names, X, y = getTrainingData(train_pathList, f)

#Fit the classifier
print('Fitting classifier...')
c = Classifier('Decision tree', X, y)

#Test the classifier
print('Running tests...')
print('\n---------------RESULTS---------------')
correct, incorrect, preds = 0, 0, []
test_pathList = ['noise_data/user']
for path in test_pathList:
    path += '/sound'
    init_path = pathlib.Path().cwd()
    wavfilenames = np.array([file for file in os.listdir(str(init_path) + '/' + path) if file.endswith(".wav")])
    vectors, labels = np.array([f.getVector(path + '/' + str(file)) for file in wavfilenames]), np.array([int(name.split('-')[2]) for name in wavfilenames])
    for i in range(len(wavfilenames)):
        file = wavfilenames[i]
        actual = labels[i]
        prediction = c.predict(vectors[i])
        preds += [prediction]
        print(path + '/' + str(file) + ': ' + getEmotion(actual) + ' | ' + getEmotion(prediction))
        if actual == prediction:
            correct += 1
        else:
            incorrect += 1

print("No. correct: " + str(correct))
print("No. incorrect: " + str(incorrect))
print("Accuracy: " + str(round(100 * correct / (incorrect + correct), 2)) + '%')
matrix = confusion_matrix(labels, np.array(preds))
print("Confusion matrix:\n" + str(matrix))