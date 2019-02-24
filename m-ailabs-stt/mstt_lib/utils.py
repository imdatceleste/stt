# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
"""
This utility module contains functions that supports for the tool (End to end multi language speech recognition using CTC Cost model)
Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
Copyright 2015-2016, Baidu USA LLC.

Based on code from Deep Speech 2 (End-to-End Speech Recognition in English and Mandarin) 
and its implementation on Github. https://github.com/baidu-research/ba-dls-deepspeech
"""

import os 
import io
import logging
import json
import numpy as np
from numpy.lib.stride_tricks import as_strided

def receive_audio_info_from_description_file(desc_file, filename):
    """ 
    Loads training audio files
    Params:
        desc_file (str):  Path to a JSON-line file that contains labels and paths to the audio files
        filename (str): Test file name that is searched in the description file
    Returns:
        audio_data (dict): audio information - file path, duration, label
    """
    print('Searching {} in description file: {} '.format(filename, desc_file))
    audio_paths, durations, texts = [], [], []
    try:
        with io.open(desc_file, encoding='utf-8') as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                spec = json.loads(json_line)
                audio_data={}
                if os.path.basename(spec['key']) == filename:
                    audio_data['key'] = spec.get('key')
                    audio_data['duration'] = float(spec.get('duration'))
                    audio_data['text'] = spec.get('text')
                    print('{} file was found in the description file'.format(filename))
                    return audio_data
            print('File {} could not be found in description file: {} '.format(filename, desc_file))
            return None
    except:
        print('Description file {} does not exist or error reading the sfile'.format(desc_file))
        return None


def calculate_conv_output_length(input_length, filter_size, border_mode, stride, dilation=1):
    """ 
    Compute the length of the output sequence after 1D convolution along time. Note that this 
    function is in line with the function used in Convolution1D class from Keras.
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
        if c == ' ':
            ch = char_map['<SPACE>']
        else:
            ch = char_map.get(c)
        int_sequence.append(ch)
    return int_sequence


def configure_logging(console_log_level=logging.INFO, console_log_format=None,  file_log_path=None, file_log_level=logging.INFO, file_log_format=None, clear_handlers=False):
    """
    Setup logging. This configures either a console handler, a file handler, or both and adds them to the root logger.
    Params:
        console_log_level (logging level): logging level for console logger
        console_log_format (str): log format string for console logger
        file_log_path (str): full filepath for file logger output
        file_log_level (logging level): logging level for file logger
        file_log_format (str): log format string for file logger
        clear_handlers (bool): clear existing handlers from the root logger
    Note:
        A logging level of `None` will disable the handler.
    """
    if file_log_format is None:
        file_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    if console_log_format is None:
        console_log_format = \
            '%(asctime)s %(levelname)-7s (%(name)s) %(message)s'

    # configure root logger level
    root_logger = logging.getLogger()
    root_level = root_logger.level
    if console_log_level is not None:
        root_level = min(console_log_level, root_level)
    if file_log_level is not None:
        root_level = min(file_log_level, root_level)
    root_logger.setLevel(root_level)

    # clear existing handlers
    if clear_handlers and len(root_logger.handlers) > 0:
        print("Clearing {} handlers from root logger.".format(len(root_logger.handlers)))
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # file logger
    if file_log_path is not None and file_log_level is not None:
        log_dir = os.path.dirname(os.path.abspath(file_log_path))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(file_log_path, mode='w')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(logging.Formatter(file_log_format))
        root_logger.addHandler(file_handler)

    # console logger
    if console_log_level is not None:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(logging.Formatter(console_log_format))
        root_logger.addHandler(console_handler) 


def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0.):
    """
    Pads each sequence to the same length: the length of the longest sequence.If maxlen is provided, 
    any sequence longer than maxlen is truncated to maxlen. Truncation happens off either the beginning 
    or the end (default) of the sequence. Supports post-padding (default) and pre-padding.
    Params:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger
        than maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
        lengths: numpy array with the original sequence lengths
    """
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int32)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)

    lengths = np.asarray([len(s) for s in x], dtype=np.int32)

    return x, lengths
