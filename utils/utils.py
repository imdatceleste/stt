# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
utils module

End to end multi language speech recognition using CTC Cost model
Copyright (c) 2019 Imdat Solak
Copyright 2015-2016, Baidu USA LLC.

This model is based on the paper of https://arxiv.org/pdf/1512.02595.pdf
(Deep Speech 2: End-to-End Speech Recognition in English and Mandarin) and its implementation on Github.
https://github.com/baidu-research/ba-dls-deepspeech

Spectrogram and mfcc implementation is based on Haytham Fayek's blog: 
http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

All rights reserved.

Implemented with Tensorflow

"""

import os
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from numpy.lib.stride_tricks import as_strided
from scipy.fftpack import fft, dct

logger = logging.getLogger(__name__)


def create_MFCC(filter_banks, num_ceps=12):
    '''
    Mel-frequency Cepstral Coefficients (MFCCs)
    Params:
        filter_banks: a numpy array (nFFT x numOfFilters)
    Returns:
        mfcc: 
    '''
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # According to the sample keeps 2-13

    # One may apply sinusoidal liftering1 to the MFCCs to de-emphasize higher MFCCs which has been claimed to improve  speech recognition in noisy signals.
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*

    # To balance the spectrum and improve the Signal-to-Noise (SNR), we can simply subtract the mean of each coefficient from all frames.
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def create_spectrogram(audio, sample_rate, window_size = 0.025, step_size = 0.01, pre_emphasis = 0.97):
    """
    Computes filter banks :
    Params:
        audio:            the input signal samples
        sample_rate:      the sampling freq (in Hz)
        window_size:   the short-term window size (in msec)
        step_size:         the short-term window step (in msec)
    Returns:
        Filterbanks: a numpy array (nFFT x numOfFilters)
    """

    # The first step is to apply a pre-emphasis filter on the signal to amplify the high frequencies.
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # After pre-emphasis, we need to split the signal into short-time frames. 
    window_length, step_length = window_size * sample_rate, step_size * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized)
    window_length = int(round(window_length))
    step_length = int(round(step_length))
    num_frames = int(np.ceil(float(np.abs(signal_length - window_length)) / step_length))  # Make sure that we have at least 1 frame

    pad_audio_length = num_frames * step_length + window_length
    z = np.zeros((pad_audio_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_audio = np.append(emphasized, z) 

    indices = np.tile(np.arange(0, window_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * step_length, step_length), (window_length, 1)).T
    frames = pad_audio[indices.astype(np.int32, copy=False)]

    # After slicing the signal into frames, we apply a window function such as the Hamming window to each frame. 
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (window_length - 1))  # Explicit Implementation **
    frames *= np.hamming(window_length)

    # We can now do an NN-point FFT on each frame to calculate the frequency spectrum and then compute the power spectrum (periodogram)    
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    # Apply triangular filters, typically 40 filters, nfilt = 40 on a Mel-scale to the power spectrum to extract frequency bands. 
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bins = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # After applying the filter bank to the power spectrum (periodogram) of the signal, we obtain the spectrogram:

    # To balance the spectrum and improve the Signal-to-Noise (SNR), we can simply subtract the mean of each coefficient from all frames.
    # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8) # Test this line

    # Return mfcc
    mfcc = create_MFCC(filter_banks)

    # return mfcc
    return filter_banks
    

def convert_int_sequence_to_text_sequence(sequence, index_map):
    """ 
    Simply convert the integer sequence to text by using the index_map object
    Params:
        sequence (list(int)): integer sequence representing the characters.
        index_map (mapper object) : comes from char_map module
    Returns:
        string representing the integer sequence
    """
    return ''.join([index_map[i] for i in sequence])


def convert_text_sequence_to_int_sequence(text, char_map):
    """ 
    Simply convert the test to integer sequence by using the char_map object
    Params:
        text (string): utterance
        char_map (mapper object) : comes from char_map module
    Returns:
        integer sequence representing the text
    """
    int_sequence = []
    for c in text:
        ch = char_map.get(c)
        int_sequence.append(ch)
    return int_sequence


def configure_logger(logFileName=None):
    '''
    Setup logging. This configures either a console handler, a file handler, or both and adds them to the root logger.
    Params:
        file_log (str): full filepath for file logger output
    '''
    if logFileName is None:
        raise ValueError("Log file name could not be found!")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-7s %(message)s')

    if not logger.handlers:
        fileHandler = logging.FileHandler(logFileName, mode='a')
        fileHandler.setLevel(logging.INFO)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)


def get_device_list():
    """
    Returns:
        Available device lists on the system
    """
    cpu = '/cpu:0'
    device_list=[]
    local_GPU_list = device_lib.list_local_devices()

    for device in local_GPU_list:
        if device.device_type=='GPU':
            device_list.append(device.name)
    if not device_list:
        device_list.append(cpu)

    return device_list


def get_sparse_tuple_from(sequences):
    '''
    Create a sparse representention of a tensor.
    Params: 
        sequences: a list of lists of type int32 where each element is a sequence
    Returns: 
        A tuple with (indices, values, shape)
    '''
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape


def get_sparse_tensor_from_dense(dense):
    '''
    Create a sparse representention of a dense tensor.
    Params: 
        dense: dense representation
    Returns: 
        targets: sparse represantation
    '''
    idx = tf.where(tf.not_equal(dense, 0))
    # Use tf.shape(dense, out_type=tf.int64) instead of dense.get_shape() if tensor shape is dynamic
    targets = tf.SparseTensor(idx, tf.gather_nd(dense, idx), tf.shape(dense, out_type=tf.int64))

    return targets

def pad_zeros(sequences):
    '''
    Pads each sequence to the same length: the length of the longest sequence.
    Params:
        sequences (numpy array 3D): mini batch sequence
    Returns
        result (numpy array 3D): padded array
        lengths (numpy array 1D): sequence length vector
    '''
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)
    maxlen = np.max(lengths)

    result = np.zeros(shape=(sequences.shape[0], maxlen, sequences.shape[2]), dtype=np.float32)

    result[:sequences.shape[0],:sequences.shape[1],:sequences.shape[2]] = sequences

    return result, lengths


def calculate_conv_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ 
    Compute the length of the output sequence after 1D convolution along time. Note that this function is in line with the 
    function used in Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


