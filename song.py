#!/usr/bin/python

from synth import Sample, Device, sin, tri, squ, Twopi, note_freq
from time import sleep

dev = Device()

#click = Sample(fn=tri(2000, 10000), domain=dev.time_range(0, .001875))
#dev.loop(click, 1)
#scale = dev.create_scale(sin(1.0/Twopi, 1.0), ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4'], domain=10)
#dev.play(scale['A4'])

import math
s0 = Sample.from_function(math.sin, note_freq('C'), 6000.0, domain=range(10000))
s1 = Sample.from_function(math.sin, note_freq('D'), 6000.0, domain=range(10000))
s2 = Sample.from_function(math.sin, note_freq('F'), 6000.0, domain=range(10000))
#s.plot()
#dev.play(s0)
#dev.play(s1)
#dev.play(s2)
dev.loop(s0, 1)
dev.loop(s1, 2)
dev.loop(s2, 4)

dev.start()
sleep(5.0)

dev.close()

