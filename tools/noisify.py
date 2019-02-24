# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
import io
import getopt
import time
import numpy as np
import soundfile as sf
from scikits.samplerate import resample
import codecs
import random
random.seed(42)

"""
noisify.py:
    Adds noise to a given audio file.
    Input is metadata.csv-directory, noise-file and noise-percentage
    If the noise-file is longer than the audio-file, it will choose randomly from
    the noise-file a segment that is then added to the audio-file
    Random-SEED is set to 42, in order to have reproducible results.

    It assumes the original wav-files to be in <metadata.csv>-dir/wavs
    It creates new wavs in <metadata.csv>-dir/wavs/<noise_filename> (w/o ext) in lower-case

Copyright (c) 2018 MUNICH ARTIFICIAL INTELLIGENCE LABORATORIES GmbH
All rights reserved.

Created: 2018-02-16 12:30 CET, KA
Updated: 2018-02-24 18:00 CET, ISO
    - Fixed indent
    - Simplified usage
    - switched to our metadata.csv-format
Updated: 2018-02-26 08:20 CET, ISO
    - Added support for random noise selection
"""

def resample_noise_file(sr_audio, sr_noise, data_noise):
    start1 = time.time()
    if sr_noise != 0:
        resampling_rate = float(sr_audio) / float(sr_noise)
        data_noise = resample(data_noise, resampling_rate, "sinc_best").astype(np.int16)
    end1 = time.time()
    # print("0:", end1-start1)
    return data_noise


def check_noise(filename):
    try:
        audio_info = sf.info(filename)
        data_noise, sr_noise = sf.read(filename, dtype='int16')
    except:
        return False, None, None

    if audio_info.format != 'WAV':
        print("Invalid noise file format {} for the file:{}".format(audio_info.format, fileName))
        return False, None, None
    elif audio_info.channels >= 2:
        data_noise = data_noise[:, 0]
        print("Noise file has two channel and kept just first channel...")
    elif audio_info.subtype != 'PCM_16':
        print(type(audio_info.subtype), audio_info.subtype)
        print("Samples must be {} for the file:{}".format(audio_info.subtype, fileName))
        data_noise = data_noise.astype(np.int16)

    return True, data_noise, sr_noise


def create_noise(new_audio_filename, audio_filename, data_audio, data_noise, sr_audio, noise_percent):
    # min_len = min(len(data_audio), len(data_noise))
    l_audio = len(data_audio)
    l_noise = len(data_noise)
    if l_audio <= l_noise:
        # We want to extract randomly a part of noise data if the noise data is 
        # larger than the audio-file
        noise_start = random.randint(0, l_noise - l_audio - 1)
        data_noise = data_noise[noise_start:noise_start + l_audio]
    else:
        new_noise = np.empty((0,))
        for i in range(int(l_audio/l_noise)+1):
            new_noise = np.concatenate((new_noise, data_noise), axis=0)
        data_noise = new_noise[:l_audio]
      # data_noise = np.pad(data_noise, (0, l_audio - l_noise%l_audio), 'constant')
    noised = (data_audio[:] * (1 - noise_percent)) + (data_noise[:] * noise_percent)
    noised = noised.astype(np.int16)
    sf.write(new_audio_filename, noised, sr_audio)


def noisify(meta_filename, noise_filename, noise_percent):
    is_noise, data_noise, sr_noise_orig = check_noise(noise_filename)
    if is_noise:
        noise_name = os.path.splitext(os.path.basename(noise_filename))[0].lower()
        print('Using noise-name: \'{}\''.format(noise_name))
        sr_noise = sr_noise_orig
        source_dir = os.path.dirname(meta_filename)
        wavs_dir = os.path.join(source_dir, 'wavs')
        out_dir = os.path.join(source_dir, 'wavs', noise_name)
        meta_outname = os.path.join(source_dir, 'metadata-{}.csv'.format(noise_name))
        meta_data = codecs.open(meta_filename, 'r', 'utf-8').readlines()
        num_entries = len(meta_data)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_meta = []
        for i, line in enumerate(meta_data):
            filename, orig_text, clean_text = line.strip().split('|')
            print('{:-7d} /{:-7d}: {}.wav{}'.format(i, num_entries, filename, ' '*30), end='\r')
            if filename.startswith('$'):
                # Filename is an absolute path...
                filename = filename[1:]
                infile = os.path.join(source_dir, filename + '.wav')
            else:
                infile = os.path.join(wavs_dir, filename + '.wav')
            outfile = os.path.join(out_dir, filename + '.wav')
            if os.path.exists(infile):
                data_audio, sr_audio = sf.read(infile, dtype='int16')
            else:
                data_audio = None
                sr_audio = None
            if data_audio is not None:
                if sr_audio != sr_noise:
                    noise_data = resample_noise_file(sr_audio, sr_noise, data_noise)
                    sr_noise = sr_audio
                create_noise(outfile, infile, data_audio, noise_data, sr_audio, noise_percent)
                out_meta.append(u'{}/{}|{}|{}'.format(noise_name, filename, orig_text, clean_text))
        print('\nSaving new metadata... ', end='')
        sys.stdout.flush()
        outf = codecs.open(meta_outname, 'w', 'utf-8')
        for l in out_meta:
            print(l, file=outf)
        outf.close()
        print('done')
        print('Added meta-file: {}'.format(meta_outname))
    else:
        print('Invalid Noise data. Exiting!')


def usage():
    print('Missing or Wrong Parameters.')
    print('Usage:') 
    print('  python noisify.py -m <meta-path> -n <noise-file> -N <noise-name> -p <percent>')
    print('      -m <metadata.csv-path>   The full path to the metadata.csv-file')
    print('      -n <noise_file>           Full path to the noise file')
    print('      -p <percent>              Noise percentage (0.0 < p < 1.0)')
    sys.exit(1)


if __name__ == '__main__':
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'm:n:p:', ['meta', 'noise_file', 'percent'])
    except getopt.GetoptError:
        usage()

    meta_filename = None
    noise_filename = None
    noise_percent = 0.0

    for opt, arg in options:
        if opt in ['-m', '--meta']:
         meta_filename = arg
        elif opt in ('-n', '--noise_file'):
         noise_filename = arg
        elif opt in ('-p', '--percent'):
            noise_percent = float(arg)
        elif opt in ("-h", "--help"):
            usage()

    if meta_filename is not None and noise_filename is not None and (0 < noise_percent < 1):
        noisify(meta_filename, noise_filename, noise_percent)
    else:
        usage()
