import numpy as np
import pylab
from scipy.io.wavfile import write
import os

# sampling rate
Fs = 44100.0

# play length
tlen = 1 # s
Ts = 1/Fs # sampling interval
t = np.arange(0, tlen, Ts) # time array

# generate signal
sin_freq = 494 # Hz
signal = np.sin(2*np.pi*sin_freq*t)

# generate noise
noise_flag = False
if noise_flag:
    noise = np.random.uniform(-1, 1, len(t)) * 0.1
else:
    noise = 0

# signal + noise
signal_n = signal + noise

# fft
signal_f = np.fft.fft(signal_n)
freq = np.fft.fftfreq(len(t), Ts)

# plot
pylab.plot(freq, 20*np.log10(np.abs(signal_f)))
pylab.xlim(0, Fs/2)
#pylab.show()

#save as wav file
scaled = np.int16(signal_n/np.max(np.abs(signal_n)) * 32767)
write('test7.wav', 44100, scaled)