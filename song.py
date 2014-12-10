#!/usr/bin/python

# This is a sample 'song' script that uses the synth library to play a series of notes and chords.

from synth import Sample, Device, sin, tri, squ, Twopi, note_freq
from time import sleep

dev = Device()

import math
s0 = Sample.from_function(math.sin, note_freq('C'), 6000.0, domain=range(10000))  # sine wave choresponding to middle C
# domain is basically the number of samples. Default sampling rate is 16000/s, so each of these samples is 10000/16000 seconds long in time duration.
s1 = Sample.from_function(math.sin, note_freq('E'), 6000.0, domain=range(10000))  # middle D
s2 = Sample.from_function(math.sin, note_freq('G'), 6000.0, domain=range(10000))  # middle G

dev.loop(s0, 1)  # plays C every beat
dev.loop(s1, 2)  # plays E every 2 beats
dev.loop(s2, 4)  # plays G every 4 beats
# Default time signature is 60 bpm

dev.start()  # open an audio device and start playing
sleep(5.0)   # let the loops run for 5 seconds
dev.close()  # close the audio device

