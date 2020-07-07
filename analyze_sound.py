import os
import glob

import scipy
import scipy.io.wavfile
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np

def plot_wav_fft(wav_filename, desc=None, trans=True):
    #plt.clf()
    fig = plt.figure(figsize=(6, 4))
    sample_rate, X = scipy.io.wavfile.read(wav_filename)
    spectrum = fftpack.fft(X)
    freq = fftpack.fftfreq(len(X), d=1.0 / sample_rate)

    ax1 = fig.add_subplot(2,1,1)
    num_samples = 300.0
    ax1.set_xlim(0, num_samples / sample_rate)
    ax1.set_xlabel("time [s]")
    ax1.set_title(desc or wav_filename)
    ax1.plot(np.arange(num_samples) / sample_rate, X[:int(num_samples)])
    ax1.grid(True)

    if trans:
        ax2 = fig.add_subplot(2,1,2)
        ax2.set_xlim(0, 24000)
        ax2.set_xlabel("frequency [HZ]")
        ax2.set_xticks(np.arange(6) * 4000)
        if desc:
            desc = desc.strip()
            fft_desc = desc[0].lower() + desc[1:]
        else:
            fft_desc = wav_filename
        ax2.set_title("FFT of %s" % fft_desc)
        print(len(freq))
        print(len(spectrum))
        ax2.plot(freq, abs(spectrum), linewidth=2)
        ax2.grid(True)
        plt.tight_layout()

    plt.show()

def plot_wav_fft_demo():
    '''
    plot_wav_fft("./test_sound/sample_record_sound/test1_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test2_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test3_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test4_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test5_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test6_iphone.wav")
    plot_wav_fft("./test_sound/sample_record_sound/test7_iphone.wav")
    '''
    plot_wav_fft("./test_sound/sample_record_sound/test_21000hz_iphone.wav")
    
plot_wav_fft_demo()
