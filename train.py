# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import random
random.seed(42) # For reproducibility

import tensorflow as tf
import os
from tensorflow.python import debug as tf_debug

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import numpy as np
import logging 
import time
import getopt
import codecs
import gc

from utils import convert_int_sequence_to_text_sequence, configure_logger
from data import DataGenerator
from utils import get_language_chars, check_language_code

"""
train module

End to end multi language speech recognition using CTC Cost model
Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE GmbH
Copyright 2015-2016, Baidu USA LLC.

This model is based on the paper of https://arxiv.org/pdf/1512.02595.pdf
(Deep Speech 2: End-to-End Speech Recognition in English and Mandarin) and its implementation on Github.
https://github.com/baidu-research/ba-dls-deepspeech

All rights reserved.

Implemented with Tensorflow

"""

"""
Module train
This modul trains the model with the selected language audio files in mini batches. 
Notes:
    File formats must be in raw format (.wav file). Audio must be sampled 16000 sample and just 1 channel data is accepted.
"""

logger = logging.getLogger(__name__)


def usage():
    print('Missing or Wrong Parameters.')
    print('Usage: train.py ...')
    print('Options:')
    print('   -e --epoch <num-epochs>         Number of training epochs(int: 100)')
    print('   -b --batch_size <batch-size>    Mini batch size for each iteration of the train (int; 32)')
    print('   -i --ilog <ilog_num>            Log decoded speech-to-text sample every this many steps (int; 100)')
    print('   -r --restore <checkpoint>       Checkpoint to restore (string; none)')
    print('   -d --data_dir <train-data-dir>  Location of training data (string; ./data)')
    print('   -g <gru_layer>                  How many gru layers (int; 3)')
    print('   -m <model_type>                 Model Type (int; 1)')
    print('   -l <language>                   Language (str, 1 of [de_DE, en_US]; en_US)')
    print('   -c <checkpoint-freq>            Save every this checkpoints (int; 1000)')
    print('   -G                              Multi-GPU-Training TRUE')
    print('   -M <model-dir>                  Model Root dir to save models to... (str; ./trained-models/)')
    print()
    sys.exit(1)


try:
    options, arguments = getopt.getopt(sys.argv[1:], 'he:b:i:r:d:g:m:c:l:M:G', ['help', 'epoch', 'batch_size', 'ilog', 'restore', 'data_dir', 'gru_layer', 'model_type', 'ckpt_freq', 'language', 'model_root'])
except getopt.GetoptError:
    usage()
num_epochs = 100
mini_batch_size = 32
iterlog = 100
restore_ckpt = None
data_dir = 'data'
checkpoint_freq = 1000
model_type = 1
gru_layer = 3
language = None
model_root = './trained-models/'
multi_gpu = False

for opt, arg in options:
    if opt in ('-e', '--epoch'):
        num_epochs = int(arg)
    elif opt in ('-b', '--batch_size'):
        mini_batch_size = int(arg)
    elif opt in ('-i','--ilog'):
        iterlog = int(arg)
    elif opt in ('-r','--restore'):
        restore_ckpt = arg
    elif opt in ('-d', '--data_dir'):
        data_dir = arg
    elif opt in ('-g', '--gru_layer'):
        gru_layer = int(arg)
    elif opt in ('-m', '--model_type'):
        model_type = int(arg)
    elif opt in ('-l', '--language'):
        language = arg
    elif opt in ('-c', '--ckpt_freq'):
        checkpoint_freq = int(arg)
    elif opt in ('-h', '--help'):
        usage()
    elif opt in ('-M', '--model_root'):
        model_root = arg
    elif opt == '-G':
        multi_gpu = True
        import horovod.tensorflow as hvd
        hvd.init()
    else:
        assert False, "unhandled option"

def write_history(model_dir, epoch_no, step_in_epoch, current_loss, best_loss, current_global_step, saved_path):
    history_file = os.path.join(model_dir, 'history.log')
    hf = codecs.open(history_file, 'a+', 'utf-8')
    print('{}|{}|{}|{}|{}|{}|{}'.format(time.time(), epoch_no, step_in_epoch, current_loss, best_loss, current_global_step, saved_path), file=hf)
    hf.close()


def train_model(language, data_dir, model_type=1, gru=3, num_epochs=100, mini_batch_size=32, iterlog=20, cp_freq=1000, restore_path=None, model_root = './trained-models/', multi_gpu=False):

    # Check language is supported
    if not check_language_code(language):
        raise ValueError("Invalid or not supported language code!")

    # Check description file exists
    if not os.path.exists(data_dir):
        raise ValueError("Description file does not exist!'")

    # Check valid model is selected
    if model_type == 1:
        from models import model_conv1_gru as model
    elif model_type == 2:
        from models import model_conv2_gru as model
    else:
        raise ValueError("No valid model selected!")

    # Create model directories
    model_name = model.__name__ + str(gru)
    model_dir = os.path.join(model_root, model_name)
    if multi_gpu:
        my_gpu_rank = hvd.local_rank()
        num_gpus = hvd.size()
    else:
        my_gpu_rank = 0
        num_gpus = 1

    if not multi_gpu or my_gpu_rank == 0:
        if not os.path.exists(model_root):
            os.makedirs(model_root)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # Configure logging
    configure_logger(logFileName=os.path.join(model_root, 'training.log'))

    print('Loading data...')
    # Load char_map, index_map and gets number of classes
    char_map, index_map, nb_classes = get_language_chars(language)

    # Prepare the data generator. Load the JSON file that contains the dataset
    datagen = DataGenerator(char_map=char_map, multi_gpu=multi_gpu)
    # Loads data limited with max duration. returns number of iterations.
    steps_per_epoch = datagen.load_data(data_dir, minibatch_size=mini_batch_size, max_duration=20.0)
    print('Building Model...')
    
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create input placeholders for CTC Cost feeding and decoding feeding
    with tf.name_scope('inputs'):
        # Audio inputs have size of [batch_size, max_stepsize, num_features]. But the batch_size and max_stepsize can vary along each step
        # inputs = tf.placeholder(tf.float32, [None, None, 161], name='inputs') # spectrogram version
        inputs = tf.placeholder(tf.float32, [None, None, 40], name='inputs') # filterbank version. 40 shows number of filters.s
        # inputs = tf.placeholder(tf.float32, [None, None, 12], name='inputs') # mfcc version. 12 shows number of ceps.
        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        # We define the placeholder for the labels. Here we use sparse_placeholder that will generate a SparseTensor required by ctc_loss op.
        targets = tf.sparse_placeholder(tf.int32, name='targets')

    # Create model layers
    logits = model(inputs, nb_classes, gru)
    logits = tf.transpose(logits, perm=[1, 0, 2])

    # Compute the CTC loss using either TensorFlow's "ctc_loss". Then calculate the average loss across the batch
    with tf.name_scope('loss'):
        total_loss =  tf.nn.ctc_loss(inputs=logits, labels=targets, sequence_length=seq_len, ignore_longer_outputs_than_inputs=True)
        avg_loss = tf.reduce_mean(total_loss, name="Mean")

    # Adam Optimizer has preferred for the performance reasons to optimize the weights
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)
        if multi_gpu:
            optimizer = hvd.DistributedOptimizer(optimizer)
        train_op = optimizer.minimize(avg_loss, global_step=global_step)
        # optimizer = tf.train.MomentumOptimizer(learning_rate= 2e-4, momentum=0.99, use_nesterov=True).minimize(avg_loss)

    # Beam search decodes the mini-batch
    with tf.name_scope('decoder'):
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=100, top_paths=1, merge_repeated=False)
        # Option 2: tf.nn.ctc_greedy_decoder (it's faster but give worse results)
        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], name="SparseToDense", default_value=-1)

    # The Levenshtein (edit) distances between the decodings and their transcriptions "distance"
    with tf.name_scope('distance'):
        distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), targets, name="edit_distance")
        # The accuracy of the outcome averaged over the whole batch ``accuracy`
        ler = tf.reduce_mean(distance, name="Mean")

    config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    init = tf.global_variables_initializer()
    total_steps  = num_epochs * steps_per_epoch
    if multi_gpu:
        config.gpu_options.allow_growth = False
        config.gpu_options.visible_device_list = str(my_gpu_rank)
        bcast = hvd.broadcast_global_variables(0)
        # Normally, we would divide the num_steps by hvd.size() 
        # But our DataGen has already done that, so we don't need it here
        # num_steps  = num_steps // hdv.size() + 1
    else:
        bcast = None
    print('Training...')
    iterator = None
    best_cost = 1e10
    lbest_cost = best_cost
    saver = tf.train.Saver(max_to_keep=50)
    session = tf.Session(config=config)
    with session.as_default():
        history_file = os.path.join(model_dir, 'history.log')
        if os.path.exists(history_file):
            entries = codecs.open(history_file, 'r', 'utf-8').readlines()
            last_entry = entries[-1].strip()
            _, h_epoch, h_epoch_step, h_loss, h_best_cost, h_cgs, h_ckptfile = last_entry.split('|')
            saver.restore(session, save_path=tf.train.latest_checkpoint(model_dir))
            epoch = int(h_epoch)
            current_epoch_step = int(h_epoch_step) + 1
            current_global_step = int(h_cgs) + 1
            remaining_epoch_steps = max(0, steps_per_epoch - current_epoch_step)
            best_cost = float(h_best_cost)
        else:
            epoch = 1
            current_epoch_step = 1
            current_global_step = 1
            remaining_epoch_steps = steps_per_epoch
            init.run()
            if bcast is not None:
                bcast.run()
        tf.get_default_graph().finalize()
        while epoch <= num_epochs:
            if epoch == 1:
                iterator = datagen.iterate_train(mini_batch_size=mini_batch_size, sort_by_duration=True, shuffle=False, max_iters=remaining_epoch_steps)
            else:
                iterator = datagen.iterate_train(mini_batch_size=mini_batch_size, sort_by_duration=False, shuffle=True, max_iters=remaining_epoch_steps)
            while current_epoch_step < steps_per_epoch:
                b_perc = int(float(current_epoch_step) / float(steps_per_epoch) * 100.0)
                inputs, out_len, indices, values, shape, labels = next(iterator)
                feed = {"inputs/inputs:0": inputs, "inputs/targets/shape:0": shape, "inputs/targets/indices:0": indices, "inputs/targets/values:0": values, "inputs/seq_len:0": out_len}
                step_start_time = time.time()
                if current_global_step % iterlog == 0:
                    _, ctc_cost, cError, cDecoded = session.run([train_op, avg_loss, ler, dense_decoded], feed_dict=feed)
                    step_end_time = time.time()
                    batch_error = cError * mini_batch_size
                    if not multi_gpu or my_gpu_rank == 0:
                        for i, seq in enumerate(cDecoded):
                            seq = [s for s in seq if s != -1]
                            sequence = convert_int_sequence_to_text_sequence(seq, index_map)
                            logger.info("IT      : {}-{}".format(current_global_step, str(i + 1)))
                            logger.info("OT ({:3d}): {}".format(len(labels[i]), labels[i]))
                            logger.info("DT ({:3d}): {}".format(len(sequence), sequence))
                            logger.info('-' * 100)
                else:
                    ctc_cost, _ = session.run([avg_loss, train_op], feed_dict=feed)
                    step_end_time = time.time()
                    if not multi_gpu or my_gpu_rank == 0:
                        best_cost_str = 'N/A' if epoch <= 1 else '{:.5f}'.format(best_cost)
                        logger.info("Epoch:{:-4d}, ES:{:-6d}, GS:{:-6d}, Loss:{:.5f}, BestLoss:{}, Time:{:.3f}".format(epoch, current_epoch_step, current_global_step, ctc_cost, best_cost_str, step_end_time - step_start_time))
                # Ignore best_cost during Epoch 1 run...
                if epoch > 1 and ctc_cost < best_cost: 
                    lbest_cost = best_cost
                    best_cost = ctc_cost
                print('Epoch: {}/{}, Step: {:-6d}/{} {:-3}% -- [Loss: {:-9.5f}, BestLoss: {:-9.5f}, Time: {:.4f}]'.format(epoch, num_epochs, current_epoch_step, steps_per_epoch, b_perc, ctc_cost, best_cost, step_end_time - step_start_time), end='\r')
                sys.stdout.flush()
                # Save every 'n' steps or when find a better best_cost
                if (current_global_step % cp_freq == 0 or best_cost < lbest_cost) and my_gpu_rank == 0:
                    print('\n*** Saving checkpoint at Epoch {}, Step {} (GS: {})'.format(epoch, current_epoch_step, current_global_step))
                    lbest_cost = best_cost
                    saved_path = saver.save(session, os.path.join(model_dir, 'model'), global_step = current_global_step)
                    write_history(model_dir, epoch, current_epoch_step, ctc_cost, best_cost, current_global_step, saved_path)
                current_epoch_step += 1
                current_global_step += num_gpus
                # Let's initiate the garbage collector to maintain acceptable RAM usage
                gc.collect()
            epoch += 1
            current_epoch_step = 0
            remaining_epoch_steps = steps_per_epoch
    print('\n')


if language is not None:
    train_model(language, data_dir, model_type, gru_layer, num_epochs, mini_batch_size, iterlog, checkpoint_freq, restore_ckpt, model_root, multi_gpu)
else:
    usage()


