# -*- encoding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import threading
import soundfile
import json
from comm_utils.tcpipserver import TCPIPConnection
from inference import SpeechToText
from utils import configure_logging
import logging
from time import sleep
from threading import Event
import datetime
import StringIO
import librosa

"""
class MLSTTJob

This class is responsible for taking a tcp-connection, reading the data, converting it to
audio (and so on) and using a SpeechToText-engine to transcribe it.
It then returns the transcribed text through the TCP-Connection to the caller (not yet implemented)
"""
class MLSTTJob(threading.Thread):
    AUDIO_PROCESS_CHUNK_SIZE = 1024000000000
    UPPER_CONFIDENCE_LEVEL = 0.95
    ALLOWED_SAMPLERATE = 16000

    def __init__(self, pool_manager, caller_event, threadID, **kwargs):
        threading.Thread.__init__(self)
        self.pool_manager = pool_manager
        self.audio_stream_connection = None
        self.thread_id = threadID
        self.consumer = kwargs.get('consumer', None)
        self.nathan_instance = kwargs.get('nathan_instance', None)
        self.language = kwargs.get('language', 'en_US')
        self.model_directory = kwargs.get('model_directory', None)
        self.model_file = kwargs.get('model_file', None)
        self.frozen_model = kwargs.get('frozen_model', False)
        self.num_runs = 0
        self.logger = logging.getLogger()
        self.caller_event = caller_event
        # self.logger.info('MLSTTJob({:d}): Creating STT-Engine and loading model...'.format(self.thread_id))
        print('MLSTTJob({:d}): Creating STT-Engine and loading model...'.format(self.thread_id))
        self.stt = SpeechToText(self.language, self.model_directory, self.model_file, self.frozen_model)
        self.aggregate_audio = ''
        self.should_stop = False

    def get_caller_event(self):
        return self.caller_event

    def _process_data(self, data):
        # FOR TESTING PURPOSES ONLY!!!!
        print('::::MLSTTJob({}) - begin TRANSCRIPTION'.format(self.thread_id))
        with soundfile.SoundFile(StringIO.StringIO(data)) as sound_file:
            audio = sound_file.read(dtype='float32')
            samplerate = sound_file.samplerate

        if samplerate != self.ALLOWED_SAMPLERATE:
            audio = librosa.core.resample(audio, samplerate, self.ALLOWED_SAMPLERATE)
            samplerate = self.ALLOWED_SAMPLERATE

        transcribed, text, alternative_text, elapsed_time, flag = self.stt.transcribe_from_voice(audio, samplerate, 1)
        result = {'transcribed': transcribed, 'text': text, 'alternative_text': alternative_text, 'elapsed_time': elapsed_time, 'flag': flag}
        data_to_send = json.dumps(result)
        self.audio_stream_connection.send_data(data_to_send)
        print('::::MLSTTJob({}) - end TRANSCRIPTION'.format(self.thread_id))
        # self.logger.info('MLSTTJob({}): TRANSCRIBED TEXT = {}'.format(self.thread_id, text))
        if False:
            if confidence > self.UPPER_CONFIDENCE_LEVEL:
                print('Recognition Complete 2 "{}"@{}'.format(text, confidence))
                self.aggregate_audio = ''
            else:
                self.aggregate_audio += data
                transcribed, text, a, b, c = self.stt.transcribe_from_voice(self.aggregate_audio, 22, 1)
                if confidence2 > self.UPPER_CONFIDENCE_LEVEL:
                    print('Recognition Complete 2 "{}"@{}'.format(text2, confidence2))
                    self.aggregate_audio = ''

    def set_tcp_connection(self, tcpconnection):
        self.audio_stream_connection = tcpconnection

    def run_count(self):
        return self.num_runs

    def _process_request(self):
        if self.audio_stream_connection is None:
            print('Missing audio-stream connection, exiting.')
            self.pool_manager.job_canceled(self)
        else:
            data_chunk = ''
            data_length = -1
            for data in self.audio_stream_connection:
                if data is not None:
                    if data_length == -1:
                        data_length = int(data[:8])
                        data = data[8:]
                    bytes_to_add_to_data_chunk = min(len(data), self.AUDIO_PROCESS_CHUNK_SIZE - len(data_chunk))
                    data_chunk += data[:bytes_to_add_to_data_chunk]
                    if len(data_chunk) == self.AUDIO_PROCESS_CHUNK_SIZE or len(data_chunk) == data_length:
                        self._process_data(data_chunk)
                        data_chunk = data[bytes_to_add_to_data_chunk:]
                else:
                    if len(data_chunk) > 0:
                        self._process_data(data_chunk)
                    break
            self.num_runs += 1
            self.aggregate_audio = ''
            self.audio_stream_connection.close()
            self.pool_manager.job_finished(self)
        self.audio_stream_connection = None

    def please_die(self):
        self.should_stop = True

    def run(self):
        while not self.should_stop:
            self.caller_event.wait()
            self.caller_event.clear()
            self._process_request()


"""
class MLSTTJobPoolManager
This class manages a pool of jobs. This is required because creating a new job
is time-consuming as it needs to load the whole model into memory...

Written: 2017-11-16 10:09 CET, ISO
"""
class MLSTTJobPoolManager(threading.Thread):
    def __init__(self, configuration, caller_event, **kwargs):
        threading.Thread.__init__(self)
        self.caller_event = caller_event
        if self.caller_event is None:
            print('ERROR: no caller_event defined. exiting')
            sys.exit(1)
        self.callee_event = Event()
        self.callee_event.clear()
        configure_logging()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.min_jobs = int(configuration.get('min_jobs', 2))
        self.max_jobs = int(configuration.get('max_jobs', 4))
        self.low_water_mark = int(configuration.get('low_water_mark', 1))
        self.max_runs_per_job = int(configuration.get('max_runs_per_job', 5))
        self.language = kwargs.get('language', 'en_US')
        self.model_directory = kwargs.get('model_directory', None)
        self.model_file = kwargs.get('model_file', None)
        self.model_is_frozen = int(configuration.get('frozen', 0)) == 1
        if self.model_directory is None:
            print('model_directory was not provided, exiting.')
            sys.exit(1)
        if self.model_file is None:
            print('model_file was not provided, exiting.')
            sys.exit(1)
        self.free_pool = []
        self.busy_pool = []
        self.new_connection = None
        self.last_call_return_value = (False, False)
        self._update_free_pool()

    def _update_free_pool(self):
        if len(self.free_pool) + len(self.busy_pool) > self.max_jobs:
            return False

        if len(self.free_pool) < self.min_jobs and len(self.free_pool) < self.low_water_mark:
            self.logger.info('MLSTTJobPoolManager:: Creating new Jobs...')
            for i in range(self.min_jobs - len(self.free_pool)):
                # self.logger.info('MLSTTJobPoolManager:: CREATING JOB NO {:d} ##########'.format(i))
                print('MLSTTJobPoolManager:: CREATING JOB NO {:d} ##########'.format(i))
                thisJobEvent = Event()
                newJob = MLSTTJob(self, thisJobEvent, len(self.free_pool) + len(self.busy_pool), language=self.language, model_directory=self.model_directory, model_file=self.model_file, frozen_model=self.model_is_frozen)
                self.free_pool.append((newJob, thisJobEvent))
                newJob.start()
        return True

    def _handle_new_connection(self):
        """
        Hands over the tcp-connection to a new MultiThreading-Job
        Fails under following conditions:
            - No free jobs left and it cannot create new free jobs
            - The TCP-connection handed is empty

        Returns two bool Values:
            1stBool, 2ndBool

        1stBool: indicates whether the call worked or not
        2ndBool: indicates whethher it can handle additional jobs or not
        """
        if self.new_connection is not None:
            tcpconnection = self.new_connection
            self.new_connection = None
            if len(self.free_pool) == 0:
                if not self._update_free_pool():
                    self.last_call_return_value = (False, False)
                    return
            newJobData = self.free_pool[0]
            newJob = newJobData[0]
            newJobEvent = newJobData[1]
            self.free_pool.remove(newJobData)
            newJob.set_tcp_connection(tcpconnection)
            self.busy_pool.append(newJobData)
            newJobEvent.set()
            if len(self.busy_pool) == self.max_jobs:
                self.last_call_return_value = (True, False)
                return
            else:
                self.last_call_return_value = (True, True)
                self._update_free_pool()
                return
        self.last_call_return_value = (False, len(self.free_pool) > 0 and len(self.busy_pool) < self.max_jobs)
        return

    def job_canceled(self, whichJob):
        """
        Currently the same as job_finished
        """
        self.job_finished(whichJob)

    def job_finished(self, whichJob):
        """
        Method called when a Job is finished. It will then be moved from the
        busy_pool to the free_pool UNLESS its run-count has exceeded maximum number
        of runs per job (to protect against memory-leaks)
        """
        event = whichJob.get_caller_event()
        jobData = (whichJob, event)
        if jobData in self.busy_pool:
            self.busy_pool.remove(jobData)
            if whichJob.run_count() < self.max_runs_per_job:
                self.free_pool.append(jobData)
            else:
                whichJob.please_die()

    def set_new_connection(self, tcpconnection):
        self.new_connection = tcpconnection

    def last_call_result(self):
        retval = self.last_call_return_value
        self.last_call_return_value = (False, False)
        return retval

    def get_callee_event(self):
        return self.callee_event

    def can_handle_more_requests(self):
        return len(self.free_pool) > 0 and len(self.busy_pool) < self.max_jobs

    def run(self):
        while True:
            self.caller_event.wait()
            self.caller_event.clear()
            self._handle_new_connection()
            self.callee_event.set()
