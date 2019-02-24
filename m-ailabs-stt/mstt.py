# -*- encoding: utf-8 -*-
from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import ConfigParser
import logging
import getopt
from time import sleep
from threading import Event
from mstt_lib.mstt_task import MLSTTJobPoolManager
from comm_utils.tcpipserver import TCPIPListener, TCPIPConnection
import comm_utils.configreader as configreader

HEADER_SIZE = 32

def main(modelConfigFilename, serverConfigFilename):
    model_config = None
    server_config = {}
    job_config = {}
    modelConfigEntries = configreader.getConfiguration(modelConfigFilename)
    model_config = modelConfigEntries.get('model', None)
    if model_config is None:
        print('Are you kidding me?? You must have model-config in the configuration file...')
        sys.exit(1)
    else:
        language = model_config.get('model_lang', 'en_US')
        model_file = model_config.get('model_file', None)
        model_directory = os.path.dirname(os.path.realpath(modelConfigFilename))
        # model_file = os.path.join(model_directory, model_file)

    if serverConfigFilename is not None:
        serverConfigEntries = configreader.getConfiguration(serverConfigFilename)
        job_config = serverConfigEntries.get('jobs', None)
        server_config = serverConfigEntries.get('server', None)
        if server_config is None:
            server_config = {}
        if job_config is None:
            job_config = {}

    IP_ADDRESS = server_config.get('listen_ip', '127.0.0.1')
    TCP_PORT = int(server_config.get('listen_port', 6666))
    BUFFER_SIZE = int(server_config.get('buffer_size', 1024))

    caller_event = Event()
    caller_event.clear()
    jobPoolManager = MLSTTJobPoolManager(job_config, caller_event, language=language, model_directory=model_directory, model_file=model_file)
    callee_event = jobPoolManager.get_callee_event()
    jobPoolManager.start()

    print('Listening on IP: {}, port: {}'.format(IP_ADDRESS, TCP_PORT))
    myListener = TCPIPListener(IP_ADDRESS, TCP_PORT, HEADER_SIZE, BUFFER_SIZE)
    while True:
        print('Waiting for connection... ')
        myConnection = myListener.wait_for_connection()
        print('Received connection, spawning job... ')
        # This part is tricky
        # We first send the jobPoolManager a new connection to handle
        # There, this information is stored in an internal variable.
        # After that, we signal the JobPoolManager that we have a job for it
        # By setting the caller_event to 'True'
        # We then immediately wait for the JobPoolManager to signal us that it
        # has accepted and forwarded the job. This happens by the JobPoolManager setting
        # the callee_event to 'True'
        # In order to clean-up everything, the WAITING thread must clear the Event-flag
        # by setting it to 'False'
        # After that, we can then acquire the message-result calling the last_call_result
        jobPoolManager.set_new_connection(myConnection)
        caller_event.set()
        callee_event.wait()
        callee_event.clear()
        result = jobPoolManager.last_call_result()
        jobAccepted = result[0]
        canAcceptMoreJobs = result[1]
        if not jobAccepted:
            print('Oops, it seems our job-pool-manager is busy...')
            myConnection.close()
        if not canAcceptMoreJobs:
            while not canAcceptMoreJobs:
                sleep(0.5)
                canAcceptMoreJobs = jobPoolManager.can_handle_more_requests()



if __name__ == '__main__':
    modelConfigFile = None
    serverConfigFile = None
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'm:s', ['model', 'server'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in options:
        if opt in ['-m', '--model']:
            modelConfigFile = arg
        elif opt in ['-s', '--server']:
            serverConfigFile = arg

    if modelConfigFile is None:
        print('ERROR: Please provide a configuration file using -m / --model option')
        sys.exit(1)

    main(modelConfigFile, serverConfigFile)
