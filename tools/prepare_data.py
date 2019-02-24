# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

"""
Analyses the length of training data and creates a train-/test-split meta-file
Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
Created: 2018-02-23 15:00 CET, ISO
Updated: 2018-02-26 10:33 CET, ISO
         - Added support for LibriSpeech data..

All rights reserved.
"""
import getopt
import json
import os
import wave
import codecs
import string
import random


def prepare_librispeech_data(top_directory, description_file):
    full_list = None
    for root, dirs, files in os.walk(top_directory):
        for filename in filter(lambda filename: filename.endswith('.trans.txt'), files):
            labelFileName = os.path.join(root, filename)
            for line in open(labelFileName):
                split = line.strip().split()
                file_id = split[0]
                label = ' '.join(split[1:]).lower()
                audio_file = os.path.join(root, file_id)
                audio = wave.open(audio_file + '.wav')
                duration = round(float(audio.getnframes()) / audio.getframerate(), 2)
                audio.close()
                # LibriSpeech has absolute path names.
                # For our SR, we prepend this with '$' to signal that this is an ABSOLUTE PATH
                full_list.append(u'${}|{:.4f}|{}|{}'.format(audio_file, duration, label, label))
    return full_list


def prepare_ljspeech_data(data_dir, test_ratio=0.0):
    meta_fname = os.path.join(data_dir, 'metadata.csv')
    if os.path.exists(meta_fname):
        full_list = []
        print('Reading metadata.csv... ', end='')
        sys.stdout.flush()
        entries = codecs.open(meta_fname, 'r', 'utf-8').readlines()
        print('done')
        num_entries  = len(entries)
        for i, line in enumerate(entries):
            filename, orig_text, clean_text = line.strip().split('|')
            print('{:-7d} /{:-7d}: {}.wav{}'.format(i, num_entries, filename, ' '*30), end='\r')
            sys.stdout.flush()
            audio_file = os.path.join(data_dir, 'wavs', filename+'.wav')
            if os.path.exists(audio_file):
                audio = wave.open(audio_file)
                duration = round(float(audio.getnframes()) / audio.getframerate(), 2)
                audio.close()
                # LJSpeech has relative paths and our SR needs to prepend 'wavs/' before it
                # Thus, we don't add '$' at the beginning of the filename
                full_list.append(u'{}|{:.4f}|{}|{}'.format(filename, duration, orig_text, clean_text))
    return full_list


def prepare_data(data_dir, test_ratio=0.0, data_format='lj'):
    if data_format.lower() not in ['lj', 'libri']:
        print('We only support LIBRIspeech or LJspeech')
        usage()
    else:
        if data_format.lower() == 'lj':
            full_list = prepare_ljspeech_data(data_dir)
        elif data_format.lower() == 'libri':
            full_list= prepare_librispeech_data(data_dir, test_ratio)

        if test_ratio > 0.0:
            random.shuffle(full_list)
            test_len = int(len(full_list) * test_ratio)
            train_len = len(full_list) - test_len
            train_list = full_list[:train_len]
            test_list = full_list[train_len:]
        else:
            train_list = full_list
            test_list = None

        if test_list is not None:
            meta_test = codecs.open(os.path.join(data_dir, 'metadata-testing.csv'), 'w', 'utf-8')
            for line in test_list:
                print(line, file=meta_test)
            meta_test.close()

        if train_list is not None:
            meta_train = codecs.open(os.path.join(data_dir, 'metadata-training.csv'), 'w', 'utf-8')
            for line in train_list:
                print(line, file=meta_train)
            meta_train.close()
        print('\nDONE')


def usage():
    print('Missing or Wrong Parameters.')
    print('Usage: ')
    print('   python prepare_data.py -d <data_dir> -f <src-data-format> [-t test_split]')
    print('      -d   data_dir for training data')
    print('      -f   source data format (lj | libri)')
    print('      -t   test_ratio (float) - should not be higher than 0.3')
    print()
    sys.exit(1)


if __name__ == '__main__':
    try:
        options, args = getopt.getopt(sys.argv[1:], 'd:t:f:', ['data_dir', 'test_ratio', 'data_format'])
    except getopt.GetoptError:
        usage()

    data_dir = None
    test_ratio = 0.0
    data_format = None
    for opt, arg in options:
        if opt in ['-d', '--data_dir']:
            data_dir = arg
        elif opt in ['-f', '--data_format']:
            data_format = arg
        elif opt in ['-t', '--test_ratio']:
            test_ratio = float(arg)
            if test_ratio > 0.3:
                print('WARNING: Test-ratio is higher than 30% (0.3)!!!!')

    if data_dir is not None and data_format is not None:
        prepare_data(data_dir, test_ratio, data_format)
    else:
        usage()

