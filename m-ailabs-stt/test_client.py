# -*- encoding: utf-8 -*-
from __future__ import print_function 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import getopt
import json
import comm_utils.configreader as configreader
from comm_utils.tcpipserver import TCPIPConnector, TCPIPConnection

def main(audioFileToSend, serverConfigFilename):
    server_config = None
    READ_SIZE = 1024
    print('Server = {}'.format(serverConfigFilename))
    if serverConfigFilename is not None:
        serverConfigEntries = configreader.getConfiguration(serverConfigFilename)
        server_config = serverConfigEntries.get('server', None)
        if server_config is None:
            print('Need a server config to connecto to. exiting')
            sys.exit(1)

    remote_ip = server_config['listen_ip']
    remote_port = int(server_config['listen_port'])
    connector = TCPIPConnector(remote_ip, remote_port)
    connection = connector.connect()
    if connection is not None:
        with open(audioFileToSend, 'rb') as f:
            data = f.read()
            if data:
                connection.send_data(data)

        rec_len = -1
        total_data = ''
        for received_data in connection:
            if received_data is not None:
                if rec_len == -1:
                    rec_len = int(received_data[:8])
                    received_data = received_data[8:]

                total_data += received_data
                if len(total_data) >= rec_len:
                    jsonD = json.loads(total_data)
                    jsonS = json.dumps(jsonD, indent=4)
                    print(jsonS)
                    connection.close()
                    break


if __name__ == '__main__':
    audioFileToSend = None
    serverConfigFile = None
    try:
        options, arguments = getopt.getopt(sys.argv[1:], 'a:s:', ['audiofile', 'server'])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in options:
        if opt in ['-a', '--audiofile']:
            audioFileToSend = arg
        elif opt in ['-s', '--server']:
            serverConfigFile = arg

    if audioFileToSend is None:
        print('Please provide an audio file to send to our little guy...')
        sys.exit(1)

    main(audioFileToSend, serverConfigFile)
