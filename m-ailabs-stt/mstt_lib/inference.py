# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import time
import json
import logging

from utils import calculate_conv_output_length, pad_sequences, convert_int_sequence_to_text_sequence
from utils import configure_logging, receive_audio_info_from_description_file
from char_map import get_language_chars, check_language_code
from filter_banks import SpectogramFeatures
"""
Class STTModelLoader
Open Issues: 
        How is the model used internally? Can we use one copy of the loaded model or do we need
        to create a single copy for EACH instance of SpeechToText class? A model uses roughly 500MiB RAM;
        We need to know exactly whether we can re-use a single copy (to save RAM) or we have to make
        copies of it (DEEP copies). We will test and see...

Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
All Rights Reserved.
"""

class STTModelLoader(object):
    def __init__(self, language, load_dir, model_file, window_size=20, step_size=10, max_freq=8000):
        if model_file == None:
            raise ValueError("Model file name does not exist!")
        elif language == None:
            raise ValueError("Language code does not exist!")
        elif load_dir == None:
            load_dir = os.getcwd()

        self.language = language
        # Check language is supported
        if not check_language_code(self.language):
            raise ValueError("Invalid or not supported language code!")

        self.load_dir = load_dir
        self.model_file = model_file
        # Check model path if exists
        full_model_file_path = os.path.join(self.load_dir, self.model_file)
        meta_file_name = full_model_file_path + ".meta"
        if not os.path.exists(meta_file_name):
            raise ValueError("META FILE {} does not exist!".format(meta_file_name))

        # We should start by creating a TensorFlow session 
        ssC = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=8)
        self.sess = tf.Session(config=ssC)
        
        # Configure logging
        configure_logging()
        self.logger = logging.getLogger()

        self.window = window_size
        self.step = step_size
        self.max_freq = max_freq

        self.logger.info("LOADING pretrained speech model ")
        # Restore model graph from previously saved model graph
        try:     
            self.saver = tf.train.import_meta_graph(meta_file_name, clear_devices=True)
            self.graph = tf.get_default_graph()
            self.logger.info("Model graph restored from file: {}".format(meta_file_name))
        except Exception as e:
            raise ValueError("Incompatible checkpoint or checkpoint file not found")

        # Create the placerholders for the inputs
        self.inputs = self.graph.get_tensor_by_name("inputs/inputs:0")
        self.seq_len = self.graph.get_tensor_by_name("inputs/seq_len:0")
        # Restore weights of the model
        with self.sess.as_default():
            self.saver.restore(self.sess, full_model_file_path)
            self.logger.info("Model weights restored from file: {}".format(full_model_file_path))

        # Load char_map and index_map
        _, self.index_map, _ = get_language_chars(self.language)

    def get_weights(self):
        return self.saver

    def get_graph(self):
        return self.graph

    def get_session(self):
        return self.sess

    def get_index_map(self):
        return self.index_map

    def get_inputs(self):
        return self.inputs

    def get_seq_len(self):
        return self.seq_len

    def get_window_size(self):
        return self.window

    def get_step_size(self):
        return self.step

    def get_max_freq(self):
        return self.max_freq

    def get_logger(self):
        return self.logger


"""
class STTFrozenModelLoader
When a tensorflow model is frozen (no additional training on that model possible anymore)
we need to load it differently than when it is not yet frozen.

We use the STTFrozenModelLoader class (which is a subclass of STTModelLoader) in such a case

WARNING: Do not call the super-class' __init__ function here

Written: 2017-11-15 08:40 CET, ISO
"""
class STTFrozenModelLoader(STTModelLoader):
    def __init__(self, language, load_dir, model_file, window_size=20, step_size=10, max_freq=8000):
        if model_file == None:
            raise ValueError("Model file name does not exist!")
        elif language == None:
            raise ValueError("Language code does not exist!")
        elif load_dir == None:
            load_dir = os.getcwd()

        self.language = language
        # Check language is supported
        if not check_language_code(self.language):
            raise ValueError("Invalid or not supported language code!")

        # Configure logging
        configure_logging()
        self.logger = logging.getLogger()

        # Check model path if exists
        self.load_dir = load_dir
        self.model_file = model_file
        full_model_file_path = os.path.join(self.load_dir, self.model_file)

        # Get the frozen graph and model data. 
        self.logger.info("LOADING pretrained speech model ")
        try:     
            self.graph = self._load_graph(full_model_file_path)
            self.logger.info("Model restored from file: {}".format(full_model_file_path))
        except Exception as e:
            raise ValueError("Incompatible checkpoint or checkpoint file not found")

        # Create the placerholders for the inputs
        self.inputs = self.graph.get_tensor_by_name("frozen/inputs/inputs:0")
        self.seq_len = self.graph.get_tensor_by_name("frozen/inputs/seq_len:0")

        self.sess = tf.Session(graph=self.graph)
        
        # Import audio parameters
        self.window = window_size
        self.step = step_size
        self.max_freq = max_freq

        # Load char_map and index_map
        _, self.index_map, _ = get_language_chars(self.language)

    def _load_graph(self, frozen_graph_filename):
        """
        Loads frozen or quantized (protobuf file (.pb)) model file. 
        Params:
            frozen_graph_filename (str): model file name
        Returns:
            graph (tf graph object): includes weights
        """
        # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
          # The name var will prefix every op/nodes in your graph
          tf.import_graph_def(graph_def, name="frozen")

        return graph


"""
Class SpeechToText
This class does the actual work.

Created: 2017-10-16 12:00 CST, KA
"""
class SpeechToText(object):
    FROZEN_MODEL = None
    CHECKPOINT_MODEL = None
    def __init__(self, language, load_dir, model_file, model_is_frozen=False):
        if model_is_frozen:
            if self.FROZEN_MODEL is None:
                self.FROZEN_MODEL = STTFrozenModelLoader(language, load_dir, model_file)
            self.loaded_model = self.FROZEN_MODEL
        else:
            if self.CHECKPOINT_MODEL is None:
                self.CHECKPOINT_MODEL = STTModelLoader(language, load_dir, model_file)
            self.loaded_model = self.CHECKPOINT_MODEL

        self.model_is_frozen = model_is_frozen
        # Get logger
        self.logger = self.loaded_model.get_logger()
        # Get Tensorflow session object 
        self.sess = self.loaded_model.get_session()
        # Get STT Model
        self.graph = self.loaded_model.get_graph()
        self.inputs = self.loaded_model.get_inputs()
        self.seq_len = self.loaded_model.get_seq_len()

        if not self.model_is_frozen:
            self.saver = self.loaded_model.get_weights()

        # Get character conversion object
        self.index_map = self.loaded_model.get_index_map()
        # Get spectogram related parameters
        self.window = self.loaded_model.get_window_size()
        self.step = self.loaded_model.get_step_size()
        self.max_freq = self.loaded_model.get_max_freq()
        # Spectogram object
        self.spectogramObj = SpectogramFeatures()

    def _prepare_data(self, audioList):
        """ 
        Featurize a minibatch of audio, zero pad them and return a dictionary
        Features is a list of (timesteps, feature_dim) arrays. Calculate the features 
        for each audio clip, as the log of the Fourier Transform of the audio
        
        Params:
            audioList (list(dict)): List of a dictionary that contains binary audio data 
            and according sample rates
        Returns:
            dict: See below for contents
        """
        features = [self.spectogramObj.featurize(audio) for audio in audioList]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        mb_size = len(features)
        # Pad all the inputs so that they are all the same length
        x = np.zeros((mb_size, max_length, feature_dim))
        for i in range(mb_size):
            feat = features[i]
            x[i, :feat.shape[0], :] = feat
        return {
            'x': x,  # (0-padded features of shape(mb_size,timesteps,feat_dim)
            'input_lengths': input_lengths,  # list(int) Length of each input
        }

    def transcribe_from_voice(self, audio, samplerate, channels, **kwargs):
        """ 
        Creates transcripted text lists from the inputted audio stream. Use the featurized audio data to feed the the STT Model and get the transcripts.
        Params:
            audiodata (numpy.ndarray): Audio data contains raw format audio data. Normally 1 channel, a one-dimensional NumPy arrar is expected. 
            If a two-dimensional NumPy array comes, it is perceived as a 2 channel data, where the channels are stored along the first dimension, 
            i.e. as columns. In this case it is converted to one-dimensional array and 1 channel data by calculating the mean of the values.
            samplerate (int): Sampling rate of the audio data per second. It must be 16000
            channels (int): Number of channel of the audio data. Must be 1 (mono). If it is 2 (stereo) then will be converted to mono by mean calculation.
        Returns:
            flag: shows the function was sucessfully returned
            transcriptList[0] (str): Trancribed voice texts. Just returns the 0'th element (first string).
            elapsed_time (float): Transcription duration
            confidence (float): always 1, because there is no ground truth to compare for this task
        """
        enter_time = time.time()

        # Control all the inputs first *****
        if audio is None:
            return False, None, -1
        elif audio.ndim > 2:
            return False, None, -1
        elif audio.ndim == 2:
            audio = np.mean(audio, 1)
        elif not isinstance(samplerate,(int)):
            return False, None, -1
        elif not samplerate in [16000]:
            return False, None, -1
        elif not isinstance(channels,(int)):
            return False, None, -1
        elif channels != 1:
            return False, None, -1

        audioList = [audio]

        for iters, batch in enumerate([self._prepare_data(audioList)]):
            acoustic_input = np.asarray(batch['x'])
            # Padding input to max_time_step of this batch
            batch_train_inputs, batch_train_seq_len = pad_sequences(acoustic_input)

            # Due to convolution, the number of timesteps of the output is different from the input length. Calculate the resulting timesteps
            batch_train_out_len = np.asarray([calculate_conv_output_length(l, 11, 'valid', 2) for l in batch_train_seq_len])
            # In order to test the model with 2D Comvolutional version remove the below two lines comments and comment the upper line
            # batch_train_conv_len = np.asarray([calculate_conv_output_length(l, 11, 'same', 2) for l in batch_train_seq_len])
            # batch_train_out_len = np.asarray([calculate_conv_output_length(l, 11, 'same', 1) for l in batch_train_conv_len])

            feed = {self.inputs: batch_train_inputs, self.seq_len: batch_train_out_len}

            # Decoding
            if self.model_is_frozen:
                dense_decoded = self.graph.get_operation_by_name("frozen/decoder/SparseToDense").outputs
            else:
                dense_decoded = self.graph.get_operation_by_name("decoder/SparseToDense").outputs
            dd = self.sess.run(dense_decoded, feed_dict=feed)

            # Convert the decoded sequence to character sequence by using char_map module function
            transcriptList = []
            for i, seq in enumerate(dd[0]):
                seq = [s for s in seq if s != -1]
                sequence = convert_int_sequence_to_text_sequence(seq, self.index_map)
                transcriptList.append(sequence)

        elapsed_time = time.time() - enter_time

        return True, transcriptList[0], '', elapsed_time, 1

