import librosa as lbr
import numpy as np

'''
Deal with everything to do with audio here. This takes as input a filepath to a .wav file and returns information
such as the features of that audio, the vector representation of that audio. 'version' specifies which version feature
representation the user wants to use.

Ideas for which features to extract: http://www.fon.hum.uva.nl/praat/
Library used for feature extraction: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
'''
class AudioAnalyzer:
    def __init__(self, version='1.0'):
        self.version = version

    '''
    Extract features from the .wav file whose location is specified by 'path'.
    '''
    def getFeatures(self, path):
        # e.g., features = getFeatures('noise_data/user/5.wav')
        signal, samplingRate = lbr.load(path)

        # Compute MFCC features from the raw signal
        frame_ms = 30
        mfcc = lbr.feature.mfcc(y=signal, sr=samplingRate, hop_length=int(samplingRate*frame_ms/1000), n_mfcc=13)

        # And the first-order differences (delta features)
        mfcc_delta = lbr.feature.delta(mfcc)

        #TODO: More features

        return (mfcc, mfcc_delta)

    '''
    Returns a vector representation of the audio from the .wav file whose location is specified by 'path'.
    '''
    def getVector(self, path):
        if self.version == '1.0':
            mfcc, mfcc_delta = self.getFeatures(path)

            #TODO: Come up with a better vector representation

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
