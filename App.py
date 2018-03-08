from Classifier import Classifier
from DataManager import DataManager
from Microphone import Recorder
import numpy as np

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
        f.storeData(path)

#r = Recorder()
#r.recordWAV(8)

f = DataManager(version='1.0')
pathList = ['noise_data/user', 'noise_data/RAVDESS/Actor_01']

storeData(pathList, f)
names, X, y = getTrainingData(pathList, f)

#Fit the classifier
c = Classifier('SVM', X, y)

#Test on what we just trained on
correct, incorrect = 0, 0
for i in range(len(names)):
    file = names[i]
    actual = y[i]
    prediction = c.predict(X[i])
    print(str(file) + ': ' + getEmotion(actual) + ' | ' + getEmotion(prediction))
    if actual == prediction:
        correct += 1
    else:
        incorrect += 1

print('\n---------------RESULTS---------------')
print("No. correct: " + str(correct))
print("No. incorrect: " + str(incorrect))
print("Accuracy: " + str(correct / (incorrect + correct)))
