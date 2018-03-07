import pyaudio
import struct
import wave
import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import speech_recognition as sr

class Recorder(object):
    def __init__(self, buffer_size = 1024*4, bitdepth=pyaudio.paInt16, channels=1, framerate=44100):
        self.chunk = buffer_size #samples per frame, i.e., size of one buffer
        self.format = bitdepth #bit-depth is in hex
        self.channels = channels #monosound
        self.rate = framerate #samples per second. 44.1 kHz default

    def recordWAV(self, duration):
        p = pyaudio.PyAudio()
        chunk = self.chunk
        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            output=True,
            frames_per_buffer=chunk
        )

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
        for i in range(0, nchunks):
            data = stream.read(self.chunk)  # data for one frame
            # convert from hex to decimal, take every other val in array
            frames.append(data)
            data_int = np.array(struct.unpack(str(2 * self.chunk) + 'B', data), dtype='b')[::2]
            line.set_ydata(data_int)
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.show(block=False)
        print("done recording")
        stream.stop_stream()
        stream.close()
        p.terminate()
        filepath = self.save(frames, p)
        return self.transcribe(filepath)

    def save(self, frames, p):
        savepath = 'noise_data/user/'
        file = open(savepath + 'names')
        lines = file.readlines()
        int_lines = [int(line) for line in lines]
        newname = np.max(int_lines) + 1

        wf = wave.open(savepath + str(newname) + '.wav', 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        int_lines.append(newname)
        output = open(savepath + 'names', "w")
        for i in range(len(int_lines)):
            output.write(str(int_lines[i]) + "\n")
        return savepath + str(newname) + '.wav'

    def transcribe(self, path):
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

r = Recorder()
print(r.recordWAV(10))