# End-to-end Speech Recognition with Multiple Language Support

This tool is used to train our speech-recognition engine.

Currently, it supports only `de_DE`, `en_US`, and `tr_TR` as language. We will add languages over time.

It requires `python 2.7`, `tensorflow 1.4-GPU` and various other packages.

*Starting 2018-03-04 the same script `train.py` suppports single- and multi-GPU training.*

    NOTE: Multi-GPU training is still experimental, so use with care...

## Preparing Data
We support two formats for data preparation:
    
    - LJSpeech
    - LibriSpeech

You can use the tool `prepare_data.py` to prepare your original data:

    python tools/prepare_data.py -d <data-dir> -f <data-format> [-t <test-ratio>]

`data-dir` is the directory where either the `metadata.csv` resides (LJSpeech) or the root of your data-directory (LibriSpeech).

`data-format` is either `lj` or `libri`.

`test-ratio` is a float between 0.0 - 1.0. This is the ratio of final data to be reserved for testing. If it is set to `0.0` no test-data will be gerated.

The result is a `metadata-training.csv` (and `metadata-testing.csv`) in the `data-dir`.

## Training
Once you have prepared your data, you can start training:

    python train.py -d <data-dir> -l <language> -b <batch-size> [-M <model-root-dir>] [-r <checkpoint-to-restore>] [-c <checkpoint-frequency>] ...

During training, a log-file will be created `training.log`, which resides in the `model-root-dir` (default: `./trained-models`).

## MultiGPU-Training

Multi-GPU training has been added and is still experimental.

You need to have OpenMPI & Horovod installed. Then you can use this command to start multi-GPU training:


    /usr/local/bin/mpirun -np 2 -H localhost:2 -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -mca btl_tcp_if_exclude enp12s0 python train.py -d /mnt/data/iso/stt/de_DE -l de_DE -b 32 -G

The parameter `btl_tcp_if_exclude enp12s0` specifies which network-connection **not** to use for inter-process communication. This is necessary only if you have more than one network-device and you want to use only one of them.

Note: the `-d` defines where your training data resides (check your own training data)

WARNING: DO NOT FORGET the `-G`-paramter in Multi-GPU training!!!

## Optional: Adding noise to your training data
If you training data is too clean, you can generate noisified versions using the `noisify.py` from `tools`:

    python tools/noisify.py -d <data-dir> -n <noise_file> -p <noise_percentage>

We have some noise-files in `tools/noise`, which you can use. The tool will run through your `metadata.csv`-files, add noise and save the result to `data-dir/wavs/noise_file_name(w/ext)/<wav-file>`.

After successfully adding noise to all your wav-files, it will write a new `metadata.csv` named `medata-<noise>.csv` in the `data-dir`

Please be patient, this can take quite some time...

# stt
