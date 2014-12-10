
import numpy as np
from struct import pack, unpack
from threading import Lock
import re
import wave
import pyaudio
from uuid import uuid4
import math

from matplotlib import pyplot

# takes a note represented by a string e.g. 'C4#' and converts it to a frequency
# value in Hz. A letter between A-G is required as the first character in the string.
# An optional positive non-zero number may follow, indicating the scale number, and
# an optional indication of sharp or flat (# or b) may follow that. If no number is given,
# scale number 4 (containing middle C) is assumed.
def note_freq(note):
    letter_ind = {'A': 0, 'B': 2, 'C': 3, 'D': 5, 'E': 7, 'F': 8, 'G': 10}
    modifiers = {'n': 0, '#': 1, 'b': -1}
    generic_error_msg = 'expected note representation like \'A4#\' or something'
    if not isinstance(note, str):
        raise TypeError('expected string')
    if note == '' or note[0].upper() not in letter_ind.keys():
        raise ValueError(generic_error_msg)
    index = letter_ind[note[0].upper()]
    if len(note) > 1:
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if note[1] in digits:
            i = 2
            while len(note) > i and note[i] in digits:
                i += 1
            index += int(note[1:i]) * 12
            if len(note) > i:
                if note[i] in modifiers.keys():
                    index += modifiers[note[i]]
                else:
                    raise ValueError(generic_error_msg)
        elif note[1] in modifiers.keys():
            index += modifiers[note[1]] + 48 # assume scale 4 if none specified
        else:
            raise ValueError(generic_error_msg)
    else:
        index += 48 # assume scale 4 if none specified
    A0 = 27.5 # frequency of A0
    return A0 * 2.0**(index / 12.0)

Twopi = 6.28318530718 # 2*pi

# normalizes a value to be in between (low) and (low + 2pi)
def norm(t, low=0.0):
    while t < low:
        t += Twopi
    while t > low + Twopi:
        t -= Twopi
    return t

# transforms a standard repeating function (amplitude=1, frequency=2pi)
# to a function with desired frequency and amplitude
def transform(fn, frequency, amplitude, rate=16000, offset=0.0):
    a = frequency * Twopi / rate
    return lambda x: amplitude * fn(x * a + offset)

# sine wave generator (not to be confused with math.sin! This returns a sine function)
def sin(frequency, amplitude, rate=16000, offset=0.0):
    return transform(math.sin, frequency, amplitude, rate, offset)
# triangle wave generator
def tri(frequency, amplitude, rate=16000, offset=0.0):
    def tri_src(t):
        t = norm(t, Twopi / 4.0)
        t -= Twopi / 4.0
        if t < Twopi / 2.0:
            return 1.0 - 4.0 * t / Twopi
        else:
            return -1.0 + 4.0 * (t - Twopi / 2.0) / Twopi
    return transform(tri_src, frequency, amplitude, rate, offset)
# square wave generator
def squ(frequency, amplitude, rate=16000, offset=0.0):
    def squ_src(t):
        t = norm(t)
        return 1.0 if t < Twopi / 2.0 else -1.0
    return transform(squ_src, frequency, amplitude, rate, offset)

# defines an arbitrary audio sample
class Sample:

    def __init__(self, data, rate=16000):
        """Initializes a new Sample from a numpy array. Should not be used directly.
        Use Sample.from_function and Sample.from_wave.
        """
        self.data = data

    @staticmethod
    def from_function(fn, frequency=1, amplitude=1, domain=1, channels=1, rate=16000):
        """Creates a new Sample object by sampling a python function
        fn:        The function to sample. Must have natural frequency 2pi and amplitude 1.
        frequency: The frequency at which to sample the function, in Hz.
        amplitude: The amplitude by which to multiply the function, must be between -60000 and 60000.
        domain:    If domain is a number n, the function will be sampled for n periods.
                   If domain is an iterable, sample the function for all x in domain (after applying transform).
        channels:  The number of channels for this sample.
        rate:      The frame rate at which the sample should be played.
        Returns a new Sample
        """
        def frame_iterator(fn, domain):
            for x in domain:
                s = fn(x)
                for i in xrange(channels):
                    yield s
        if isinstance(domain, (int, float, long)):
            if domain <= 0:
                raise ValueError('expected a positive domain')
            end_time = int(round(domain * Twopi * rate / frequency))
            domain = xrange(end_time)
        iterator = frame_iterator(transform(fn, frequency, amplitude, rate=rate), domain)
        return Sample(np.fromiter(iterator, int).reshape((-1, channels)), rate)

    @staticmethod
    def from_wave(filename, frequency=1.0, amplitude=1.0, domain=1, channels=None, rate=16000):
        """Creates a new Sample object by sampling a python function
        filename:  The path to the wave file.
        frequency: The relative frequency at which to sample the file (output(i) = amplitude*input(frequency*i)).
        amplitude: The amplitude by which to multiply the sample (output(i) = amplitude*input(frequency*i))
        domain:    If domain is a number n, the whole file will be sampled for n loops.
                   If domain is an iterable, sample the function for all x in domain (after applying transform).
        channels:  The number of channels for this sample (may be different from the file's channels).
        rate:      The frame rate at which the sample should be played (may be different from the file's rate).
        Returns a new Sample
        """
        wf = wave.open(filename, 'r')
        print 'loaded framerate = %s' % wf.getframerate()
        scale = wf.getframerate() / float(rate)
        max_i = int(wf.getnframes() / scale)
        if domain is None:
            domain = xrange(max_i)
        def frame_generator(wf, domain):
            data = wf.readframes(wf.getnframes())
            fchannels = wf.getnchannels()
            twochans = fchannels * 2
            index_mul = scale * twochans
            for i in domain:
                x = (i % max_i) * index_mul
                if x == int(x):
                    c = unpack('<%dH' % fchannels, data[x:x + twochans])
                    for j in xrange(channels):
                        yield c[j % fchannels]
                else:
                    x_low, x_high = int(x), int(x) + twochans
                    low = unpack('<%dH' % fchannels, data[x_low:x_low + twochans])
                    high = unpack('<%dH' % fchannels, data[x_high:x_high + twochans])
                    mul = (x - x_low) / float(x_high - x_low)
                    for j in xrange(channels):
                        k = j % fchannels
                        yield low[k] + float(high[k] - low[k]) * mul
        data = np.fromiter(frame_generator(wf, domain), int, len(domain) * channels).reshape((-1, channels))
        wf.close()
        return Sample(data, rate)

    def __getitem__(self, key):
        return self.data[int(key) % len(self.data)]

    def plot(self):
        pyplot.plot(range(len(self.data)), self.data)
        pyplot.show()

class Device:
    def __init__(self, buff_size=1024, frame_rate=16000, channels=1, start=False, bpm=120):
        self.buff_size = buff_size
        self.frame_rate = frame_rate
        self.channels = channels
        self.pa = pyaudio.PyAudio()
        self.playlist = {}
        self.loops = {}
        self.i = 0
        self.lock = Lock()
        self.set_time_sig(bpm)
        if start:
            self.start()

    def _frame_generator(self, n):
        for i in xrange(n):
            self.lock.acquire()
            for uid, (sample, t) in self.playlist.items():  # update t / perge finished samples
                if t >= len(sample.data):
                    del self.playlist[uid]
                else:
                    self.playlist[uid] = (sample, t + 1)
            for uid, (sample, interval) in self.loops.items():
                if self.i % interval == 0:
                    self.playlist[uid] = (sample, 0)
            cc = reduce(lambda x, y: np.fromiter((x[i] + y[i % len(y)] for i in xrange(self.channels)), dtype=int, count=self.channels),
                        (s[t] for (s, t) in self.playlist.values()),
                        np.zeros(self.channels))
            self.lock.release()
            for j in cc:
                yield j
            self.i += 1
    def _fn(self, in_data, frame_count, time_info, status):
        data = pack('%dh' % (frame_count * self.channels), *self._frame_generator(frame_count))
        return (data, pyaudio.paContinue)

    def start(self):
        self.stream = self.pa.open(format=pyaudio.paInt16,
                                   rate=self.frame_rate,
                                   channels=self.channels,
                                   output=True,
                                   frames_per_buffer=self.buff_size,
                                   stream_callback=self._fn)
        self.needclose = True
        self.stream.start_stream()

    def set_time_sig(self, bpm=120):
        self.bpm=bpm

    def play(self, sample):
        uid = uuid4()
        self.lock.acquire()
        self.playlist[uid] = (sample, 0)
        self.lock.release()
        return uid

    def loop(self, sample, beats):
        uid = uuid4()
        self.lock.acquire()
        if isinstance(beats, list):
            for beat in beats:
                self.loops[uid] = (sample, beats * 60 * self.frame_rate / self.bpm)
        else:
            self.loops[uid] = (sample, beats * 60 * self.frame_rate / self.bpm)
        self.lock.release()
        return uid

    def time_range(self, start, stop, step=None):
        return xrange(int(round(start * self.frame_rate)),
                      int(round(stop * self.frame_rate)),
                      1 if step is None else int(round(step * self.frame_rate)))

    # takes a standard repeating function and a list of notes (see note_freq) and returns
    # a dictionary of samples taken at the associated frequencies. If domain is a range,
    # sample each note along that domain. If it's a number n, sample n periods of the function.
    # If it's None, sample 1 period of the function.
    def create_scale(self, fn, notes, domain=None, amplitude=30000):
        if domain is None:
            domain = 1
        if isinstance(domain, (int, long, float)):
            r = {}
            for note in notes:
                f = note_freq(note)
                r[note] = Sample(fn=transform(fn, f, amplitude), domain=xrange(int(round(domain * self.frame_rate / float(f)))), frame_rate=self.frame_rate, channels=self.channels)
            return r
        else:
            return {note: Sample(fn=transform(fn, note_freq(note), amplitude), domain=domain, frame_rate=self.frame_rate, channels=self.channels) for note in notes}

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

    def close(self):
        if self.stream.is_active():
            self.stop()
        if self.needclose:
            self.pa.terminate()
            self.needclose = False

