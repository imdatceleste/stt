# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import os
import io
import random
import json
import soundfile

from numpy.lib.stride_tricks import as_strided

"""
Class SpectogramFeatures
This class calculates the spectogram features of a given audio clip

Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
Copyright 2015-2016, Baidu USA LLC.

Based on code from Deep Speech 2 (End-to-End Speech Recognition in English and Mandarin) 
and its implementation on Github. https://github.com/baidu-research/ba-dls-deepspeech

Spectrogram and mfcc implementation is based on Haytham Fayek's blog: 
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html


All rights reserved.

"""

class SpectogramFeatures():

    def _create_spectrogram(self, signal, sample_rate, frame_size = 0.025, frame_stride = 0.01, pre_emphasis = 0.97):
        """
        Computes filter banks :
        Params:
            signal:       the input signal samples
            sample_rate:  the sampling freq (in Hz)
            frame_size:   the short-term window size (in msec)
            frame_stride: the short-term window step (in msec)
        Returns:
            Filterbanks: a numpy array (nFFT x numOfFilters)
        """

        # The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies.
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # After pre-emphasis, we need to split the signal into short-time frames. The rationale behind this step is that 
        # frequencies in a signal change over time, so in most cases it doesn’t make sense to do the Fourier transform 
        # across the entire signal in that we would loose the frequency contours of the signal over time. To avoid that, 
        # we can safely assume that frequencies in a signal are stationary over a very short period of time. Therefore, 
        # by doing a Fourier transform over this short-time frame, we can obtain a good approximation of the frequency 
        # contours of the signal by concatenating adjacent frames.
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        pad_signal = np.append(emphasized_signal, z) 

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # After slicing the signal into frames, we apply a window function such as the Hamming window to each frame. 
        # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
        frames *= np.hamming(frame_length)

        # We can now do an NN-point FFT on each frame to calculate the frequency spectrum, which is also called Short-Time 
        # Fourier-Transform (STFT), where NN is typically 256 or 512, NFFT = 512; and then compute the power spectrum (periodogram)    
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        # The final step to computing filter banks is applying triangular filters, typically 40 filters, nfilt = 40 on a Mel-scale 
        # to the power spectrum to extract frequency bands. The Mel-scale aims to mimic the non-linear human ear perception of sound, 
        # by being more discriminative at lower frequencies and less discriminative at higher frequencies.
        nfilt = 40
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB
        # After applying the filter bank to the power spectrum (periodogram) of the signal, we obtain the spectrogram:

        # To balance the spectrum and improve the Signal-to-Noise (SNR), we can simply subtract the mean of each coefficient from all frames.
        # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8) # Test this line

        return filter_banks
    
    def featurize(self, audio, samplerate=16000):
        """ 
        For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip (str): audio data array
            samplerate (int)= sampling rate of the audio data per second
        Returns:
            In return function calls the _spectrogram_from_audio_segment function
        """
        if audio is None:
            raise ValueError("No audio found!")
        elif not isinstance(samplerate,(int)):
            raise ValueError("Sample rate must be integer!")
        elif not samplerate in [16000]:
            raise ValueError("Sample rate should be in allowed values!")

        return self._create_spectrogram(audio, samplerate)
