from Classifier import Classifier
from DataManager import DataManager

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

#Extract data
f = DataManager(version='1.0')
#f.storeData('noise_data/RAVDESS/Actor_01')
names, X, y = f.getDataFromFile('noise_data/RAVDESS/Actor_01')
names_cum, X_cum, y_cum = names, X, y

#Fit the classifier
c = Classifier('SVM', X_cum, y_cum)

#Test on what we just trained on
for i in range(len(names_cum)):
    file = names[i]
    emotion = getEmotion(y_cum[i])
    label = c.predict(X[i])
    print('noise_data/RAVDESS/Actor_01/' + str(file) + ': ' + emotion + ' | ' + getEmotion(label))