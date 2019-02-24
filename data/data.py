# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
data module

End to end multi language speech recognition using CTC Cost model
Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
Copyright 2015-2016, Baidu USA LLC.

This model is based on the paper of https://arxiv.org/pdf/1512.02595.pdf
(Deep Speech 2: End-to-End Speech Recognition in English and Mandarin) and its implementation on Github.
https://github.com/baidu-research/ba-dls-deepspeech

All rights reserved.

Implemented with Tensorflow

"""

import json
import logging
import numpy as np
import random
import io
import librosa
import codecs
import os
import scipy.signal

from utils import create_spectrogram, convert_text_sequence_to_int_sequence, pad_zeros, calculate_conv_output_length, get_sparse_tuple_from
from utils import clean_text


logger = logging.getLogger(__name__)

"""
Class DataGenerator
This class is used to featurize audio clips, and provide them to the network for training or testing.
Usage:
    myDataGenerator = DataGenerator(char_map)
Params:
    char_map (dict): Character mapping object.
"""
class DataGenerator(object):
    def __init__(self, char_map=None, multi_gpu=False):
        self.RNG_SEED = 52
        self.char_map = char_map
        self.multi_gpu = multi_gpu
        if multi_gpu:
            import horovod.tensorflow as hvd
            self.num_gpus = hvd.size()
            self.rank = hvd.local_rank()
        else:
            self.num_gpus = 1
            self.rank = 0

    def load_data(self, data_dir, minibatch_size=16, max_duration=20.0, testing=False):
        """ 
        Loads training audio files
        Params:
            data_dir: Path to the training_data
                This script expects a 'metadata-training.csv' in this path
                and the audio files to be in 'wavs' directory there
            max_duration (float): In seconds, the maximum duration of utterances to train or test on
        """
        audio_paths, durations, texts = [], [], []
        meta_name = 'metadata-testing.csv' if testing else 'metadata-training.csv'
        desc_file = codecs.open(os.path.join(data_dir, meta_name), 'r', 'utf-8')
        for line_num, line in enumerate(desc_file):
            filename, duration, orig_text, processed_text = line.strip().split('|')
            if float(duration) <= max_duration:
                if filename.startswith('$'): # It is an absolute path
                    filename = filename[1:]
                    audio_file = os.path.join(data_dir, filename) + '.wav'
                else:
                    audio_file = os.path.join(data_dir, 'wavs', filename) + '.wav'
                ctext = clean_text(processed_text, self.char_map)
                if ctext is not None and len(ctext)>0:
                    audio_paths.append(audio_file)
                    durations.append(float(duration))
                    texts.append(ctext)
                else:
                    print('INFO: {:-6d}: Cleaning "{}" resulted in empty text...'.format(line_num, processed_text))
        desc_file.close()
        _dur, _ap, _te = DataGenerator._sort_by_duration(durations, audio_paths, texts)
        if self.multi_gpu:
            audio_paths, durations, texts = [], [], []
            for i, entry in enumerate(_dur):
                if i % self.num_gpus == 0:
                    durations.append(entry)
                    audio_paths.append(_ap[i])
                    texts.append(_te[i])
        else:
            durations = _dur
            audio_paths = _ap
            texts = _te
        self.train_audio_paths = audio_paths
        self.train_durations = durations
        self.train_texts = texts
        self.num_examples = len(durations)
        return int(np.ceil(len(durations) / minibatch_size))

    @staticmethod
    def _sort_by_duration(durations, audio_paths, texts):
        # Sort the audio files according to duration size
        return zip(*sorted(zip(durations, audio_paths, texts)))

    @staticmethod
    def _preprocess_data(x, y, texts):
        inputs, seq_len = pad_zeros(x)
        # Due to convolution, the number of timesteps of the output is different from the input length. Calculate the resulting timesteps
        out_len = np.asarray([calculate_conv_output_length(l, 11, 'valid', 2) for l in seq_len])
        maxlen= max(out_len)
        # Pad the label sequences with "blank" and truncate accordingly "maxlens". Label lengths must be smaller than or equal to the sequence length.
        labels=np.asarray([lbl[:maxlen] if len(lbl) >= maxlen else lbl  for lbl in y])
        # Creating sparse representation of the labels to feed the placeholder
        indices, values, shape = get_sparse_tuple_from(labels)
        # return inputs, out_len, indices, values, shape, texts
        return inputs, out_len, indices, values, shape, texts

    def _prepare_minibatch(self, audio_paths, texts):
        STT_SAMPLE_RATE = 16000
        """ 
        Featurize a minibatch of audio, zero pad them and return a dictionary
        
        Params:
            audio_paths (list(str)): List of paths to audio files
            texts (list(str)): List of texts corresponding to the audio files
        Returns:
            dict: See below for contents
        """
        assert len(audio_paths) == len(texts), "Inputs and outputs to the network must be of the same number"
        # Features is a list of (timesteps, feature_dim) arrays calculate the features for each audio clip, 
        # as the log of the Fourier Transform of the audio
        features = []
        for filename in audio_paths:
            audio, sample_rate = librosa.core.load(filename, mono=True, sr=STT_SAMPLE_RATE, dtype='float32')
            filter_banks = create_spectrogram(audio, sample_rate=sample_rate)
            features.append(filter_banks)
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        y = []
        label_lengths = []
        for i in range(mb_size):
            feat = features[i]
            x[i, :feat.shape[0], :] = feat
            label = convert_text_sequence_to_int_sequence(texts[i], self.char_map)
            y.append(label)
            label_lengths.append(len(label))
        return DataGenerator._preprocess_data(x, np.asarray(y), texts)

    def _iterate(self, audio_paths, texts, mini_batch_size=32, max_iters=0):
        """
        Creates minibatches given audio paths and texts

        Returns:
            minibatches given mini batch size
        """
        if max_iters != 0:
            k_iters = max_iters
        else:
            raise ValueError("Number of iterations is Zero!")
        start = 0 
        start = max(0, len(audio_paths) - (k_iters * mini_batch_size))
        for i in range(k_iters):
            minibatch = self._prepare_minibatch(audio_paths[start: start + mini_batch_size], texts[start: start + mini_batch_size])
            yield minibatch
            start += mini_batch_size

    def iterate_train(self, mini_batch_size=32, sort_by_duration=True, shuffle=False, max_iters=0, epoch_no=1):
        # Checks the sort criteria and shuffles the data accordingly. Then calls the _iterate to create the mini batches.
        if sort_by_duration and shuffle:
            shuffle = False
            logger.warn("Both sort_by_duration and shuffle were set to True. Setting shuffle to False")
        durations, audio_paths, texts = (self.train_durations, self.train_audio_paths, self.train_texts)
        if shuffle:
            temp = list(zip(durations, audio_paths, texts))
            if epoch_no > 1 and self.num_gpus > 1:
                random.seed(self.RNG_SEED) # We need to make sure that the randomization across instances is identical!
                self.RNG_SEED += 1         # But we don't want to have the same random numbers at each shuffle-call...
            random.shuffle(temp)
            # self.rng.shuffle(temp)
            durations, audio_paths, texts = zip(*temp)
        if sort_by_duration:
            durations, audio_paths, texts = DataGenerator._sort_by_duration(durations, audio_paths, texts)
        return self._iterate(audio_paths, texts, mini_batch_size, max_iters)

