import pyaudio
import struct
import wave
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr

'''
This class allows audio input from users. It records and stores audio in the .wav file format. If the record() function
crashes on the first run, simply run again and it should work.
'''
class Recorder(object):
    def __init__(self, buffer_size = 1024*4, bitdepth=pyaudio.paInt16, channels=1, framerate=44100):
        # samples per frame, i.e., size of one buffer
        self.chunk = buffer_size
        # bit-depth, i.e., range of values of pressure at a particular time. Hex by default
        self.format = bitdepth
        # monosound by default
        self.channels = channels
        # samples per second. 44.1 kHz default
        self.rate = framerate

    '''
    Calling this function records audio for 'duration' number of seconds. A pyplot is displayed to illustrate sound
    input and a file is saved at the end. Returns the text of what was said.
    '''
    def recordWAV(self, duration):
        p = pyaudio.PyAudio()
        chunk = self.chunk
        #Create audio stream
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=chunk
        )

        #Create graph representation of signal
        fig, ax = plt.subplots()
        x = np.arange(0, 2*self.chunk, 2)
        line, = ax.plot(x, np.random.rand(self.chunk))

        ax.set_ylim(-128, 127)
        ax.set_xlim(0, self.chunk)

        plt.show(block = False)
        RECORD_SECONDS = duration
        nchunks = int(RECORD_SECONDS * self.rate / self.chunk)

        frames = []
        print("recording...")
        #Optimized graph displaying so that display isn't choppy
        for i in range(0, nchunks):
            #data from one chunk
            data = stream.read(self.chunk)
            frames.append(data)
            #convert from hex to decimal, take every other val in array
            data_int = np.array(struct.unpack(str(2 * self.chunk) + 'B', data), dtype='b')[::2]
            line.set_ydata(data_int)
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.show(block=False)
        print("done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        #save the data from the recording
        filepath = self.save(frames, p)
        #return the transription of what was said
        return self.transcribe(filepath)

    '''
    Save the data from the signal in a file in the directory noise_data/user. The 'names' text file is used to keep
    track of what file names we have already used so that we don't accidentaly overwrite an old file when saving a new
    file.
    '''
    def save(self, frames, p):
        savepath = 'noise_data/user/'
        #figure out what filenames have already been used and use a new file name
        file = open(savepath + 'names')
        lines = file.readlines()
        int_lines = [int(line) for line in lines]
        distinguisher = np.max(int_lines) + 1
        print("01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised")
        emotion = input('Which emotion was that: ')

        newname = str(distinguisher) + '--' + emotion + '-'
        #save the signal data in a .wav file format
        wf = wave.open(savepath + 'sound/' + str(newname) + '.wav', 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        int_lines.append(distinguisher)
        output = open(savepath + 'names', "w")
        for i in range(len(int_lines)):
            output.write(str(int_lines[i]) + "\n")
        return savepath + 'sound/' + str(newname) + '.wav'

    '''
    Uses google API to translate audio data to text.
    '''
    @staticmethod
    def transcribe(path):
        r = sr.Recognizer()
        with sr.WavFile(path) as source:
            audio = r.record(source)  # extract audio data from the file

        try:
            text = r.recognize_google(audio)  # recognize speech using Google Speech Recognition
        except LookupError:  # speech is unintelligible
            text = "Could not understand audio"
        except sr.UnknownValueError:
            text = "Could not understand audio"
        return text