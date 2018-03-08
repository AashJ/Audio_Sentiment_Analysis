import librosa as lbr
#http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
import numpy as np

class AudioAnalyzer:
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
