import librosa as lbr
import numpy as np

'''
Deal with everything to do with audio here. This takes as input a filepath to a .wav file and returns information
such as the features of that audio, the vector representation of that audio. 'version' specifies which version feature
representation the user wants to use.

Ideas for which features to extract: http://www.fon.hum.uva.nl/praat/
More ideas for which features to extract: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.690.6309&rep=rep1&type=pdf
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

        # Compute energy of each frame
        energy = lbr.feature.rmse(y=signal, hop_length=int(samplingRate*frame_ms/1000))

        # Compute pitch using autocorrelation
        autocorrelation = lbr.autocorrelate(signal)

        # The first-order differences (delta features)
        #mfcc_delta = lbr.feature.delta(mfcc)

        # Zero-crossing rate
        #zerorate = lbr.feature.zero_crossing_rate(signal)

        # Roll-off frequency
        #rolloff = lbr.feature.spectral_rolloff(y=signal, sr=samplingRate)

        # Spectral bandwidth
        #bandwidth = lbr.feature.spectral_bandwidth(y=signal, sr=samplingRate)

        #TODO: More features

        return (mfcc, energy, autocorrelation)

    '''
    Returns a vector representation of the audio from the .wav file whose location is specified by 'path'.
    '''
    def getVector(self, path):
        if self.version == '1.0':
            mfcc, energy, autocorrelation = self.getFeatures(path)

            #TODO: Come up with a better vector representation

            vector = [0]*3000

            count = 0
            for list in mfcc:
                for i in range(len(list)):
                    if count < 1000:
                        vector[count] = list[i]
                    count += 1
                if count > 999:
                    break
            count = 1000

            for list in energy:
                for i in range(len(list)):
                    if count < 2000:
                        vector[count] = list[i]
                    count += 1
                if count > 1999:
                    break
            count = 2000

            for i in range(len(autocorrelation)):
                if count < 3000:
                    vector[count] = autocorrelation[i]
                count += 1

            count = 2999

            '''
            for list in mfcc_delta:
                for i in range(len(list)):
                    if count < 1500:
                        vector[count] = list[i]
                    count += 1
                if count > 1499:
                    break
            count = 1500

            for list in zerorate:
                for i in range(len(list)):
                    if count < 2000:
                        vector[count] = list[i]
                    count += 1
                if count > 1999:
                    break
            
            for list in rolloff:
                for i in range(len(list)):
                    if count < 2500:
                        vector[count] = list[i]
                    count += 1
                if count > 1999:
                    break

            for list in bandwidth:
                for i in range(len(list)):
                    if count < 2500:
                        vector[count] = list[i]
                    count += 1
                if count > 2499:
                    break
            '''

            return vector
        else:
            return np.array([0]*100)
