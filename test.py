# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
train module

End to end multi language speech recognition using CTC Cost model
Copyright (c) 2019 Imdat Solak
Copyright 2015-2016, Baidu USA LLC.

This model is based on the paper of https://arxiv.org/pdf/1512.02595.pdf
(Deep Speech 2: End-to-End Speech Recognition in English and Mandarin) and its implementation on Github.
https://github.com/baidu-research/ba-dls-deepspeech

All rights reserved.

Implemented with Tensorflow

"""

import random
random.seed(42) # For reproducibility

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import logging 
import getopt
import time
import re

from keras.layers import (Conv1D, Conv2D, Dense, GRU, TimeDistributed, Bidirectional, InputLayer)
from keras.layers.normalization import BatchNormalization

from utils import get_language_chars, check_language_code
from data import DataGenerator
from utils import convert_int_sequence_to_text_sequence


"""
Module create-inference

This modul creates an inference model by using multi GPU trained models. It also creates the frozen version at the same time.

Usage:
    create_inference.py <-c configuration-file>
Options:
    Usage: create_inference.py <-c configuration-file> -o <output-node-names>')
        -c --config                 <config-file>              Training configuration file')
        -o --output_node_names    <output-node-names>        In order to create the inference path frozen model needs this parameter')
"""

def test_model(language, data_dir, model_path, model_type, gru, mini_batch_size=32, test_report_file='report.log'):
    if not check_language_code(language):
        raise ValueError("Invalid or not supported language code!")

    report_file = codecs.open(test_report_file, 'w', 'utf-8')
    char_map, index_map, nb_classes = get_language_chars(language)
    datagen = DataGenerator(char_map=char_map)
    no_iter = datagen.load_data(data_dir, minibatch_size=mini_batch_size, max_duration=20.0, testing=True)
    print("Number of Iterations of Testing Data is: {}".format(no_iter))
    print("BUILDING TEST GRAPH")
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.float32, [None, None, 40], name='inputs') # filterbank version. 40 shows number of filters.s
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        targets = tf.sparse_placeholder(tf.int32, name='targets')

    from models import model_conv1_gru as model
    logits = model(inputs, nb_classes, gru)
    logits = tf.transpose(logits, perm=[1, 0, 2])
    with tf.name_scope('decoder'):
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=100, top_paths=1, merge_repeated=False)
        # Option 2: tf.nn.ctc_greedy_decoder (it's faster but give worse results)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], name="SparseToDense", default_value=-1)

    print("Loading model from checkpoint {}".format(model_path), end='')
    sys.stdout.flush()
    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    with sess.as_default():
        try:
            saver.restore(sess, save_path=model_path)
        except:
            raise ValueError("Checkpoint could not be loaded!")
        print('done')
        print('Testing...')
        iters = 1
        iter_batch = iter(datagen.iterate_train(mini_batch_size=mini_batch_size, shuffle=False, sort_by_duration=True, max_iters=no_iter))
        while iters <= no_iter:
            print('Item: {:-6d}/{}'.format(iters, no_iter), end='\r')
            inputs, out_len, indices, values, shape, labels = next(iter_batch)
            feed = {"inputs/inputs:0": inputs, "inputs/targets/shape:0": shape, "inputs/targets/indices:0": indices, "inputs/targets/values:0": values, "inputs/seq_len:0": out_len}
            iter_start = time.time()
            dd = sess.run(dense_decoded, feed_dict=feed)
            iter_end = time.time()
            for i, seq in enumerate(dd):
                seq = [s for s in seq if s != -1]
                sequence = convert_int_sequence_to_text_sequence(seq, index_map)
                print(u'{:%.3f}|{}|{}'.format(iter_end-iter_start, labels[i], sequence), file=report_file)
            iters += 1
    print('\ndone. Report is in {}'.format(test_report_file))
    report_file.close()
    return True


def usage():
    print('Missing or Wrong Parameters.')
    print('Usage: test.py -d <data-dir> -l <language> -M <model_path> ...')
    print('Options:')
    print('   -d --data_dir <test-data-dir>   Location of testing data (string; ./data)')
    print('   -l <language>                   Language (str, 1 of [de_DE, en_US]; en_US)')
    print('   -M --model_path <checkpoint>    Model to load (checkpoint) (string; none)')
    print('   -b --batch_size <batch-size>    Mini batch size for each iteration of the train (int; 32)')
    print('   -r <report_file>                Filename for the report (str; report.log')
    print('   -g <gru_layer>                  How many gru layers (int; 3)')
    print('   -m <model_type>                 Model Type (int; 1)')
    print()
    sys.exit(1)


if __name__ == '__main__':
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'he:b:d:g:m:l:M:r:', ['help', 'batch_size', 'data_dir', 'gru_layer', 'model_type', 'language', 'model_path', 'report_file'])
    except getopt.GetoptError:
        usage()
    mini_batch_size = 32
    data_dir = 'data'
    model_type = 1
    gru_layer = 3
    language = None
    model_path = None
    test_report_file = 'report.log'

    for opt, arg in options:
        if opt in ('-b', '--batch_size'):
            mini_batch_size = int(arg)
        elif opt in ('-d', '--data_dir'):
            data_dir = arg
        elif opt in ('-g', '--gru_layer'):
            gru_layer = int(arg)
        elif opt in ('-m', '--model_type'):
            model_type = int(arg)
        elif opt in ('-l', '--language'):
            language = arg
        elif opt in ('-h', '--help'):
            usage()
        elif opt in ('-M', '--model_path'):
            model_path = arg
        elif opt in ('-r', '--report_file'):
            test_report_file = arg
        else:
            assert False, "unhandled option"

    if language is not None and data_dir is not None and model_path is not None:
        _ = test_model(language, data_dir, model_path, model_type, gru_layer, mini_batch_size, test_report_file)
    else:
        usage()

