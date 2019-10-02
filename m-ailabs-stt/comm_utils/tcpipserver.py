# -*- encoding: utf-8 -*-
from __future__ import print_function 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import socket
"""
A few classes to setup a TCP/IP-server that waits for connections
and allows you to iterate of the connection's data (sequentially)

Copyright (c) 2019 Imdat Solak

Created: 2017-10-12 15:30 CET, ISO
Updated: 2018-10-13 12:00 CET, ISO
         - Added sending of data via the TCP connection
         - Renamed TCPReceive to TCPIPConnection
         - Added close_connection to TCPIPConnection
         - Added shutdown to TCPIPListener


Usage:
    First, you need to setup the tcp/ip server (listener):

    myListener = TCPIPListener(<ipv4-ipaddress>, <port> [, <header_size>])

    With that, the TCPIPListener will bind to that IP-Address and PORT.
    Next, you probably do the following in your code:

    myConnection = myListener.wait_for_connection()

    Then you will get a TCPIPConnection-object.

    Now you can iterate over the TCPIPReceiver-Object in classical Python terms:

    myReceivedData = ''
    for data in myConnection:
        if data is None:
            break
        myReceivedData += data

    If you receive 'None' as data, then the connection has been closed.


    If you want to write a multi-threaded/processed app, you would do the following:

    while True:
        myConnection = myListener.wait_for_connection()
        newProc = Process(target=<my_process_function>, args=(myConnection, ...))
        newProc.start()
        ...
"""

"""
TCPIPConnection

This is an iterable object. At every call to 'next()', it will return the next
chunk of data that it receives from the connection.

When you receive 'None' as data (from a call to 'next()'), it just means that the
connection has been closed.

YOU NEVER INSTANTIATE this class yourself. You receive an instance of this class
from the TCPIPListener object when you send it a 'wait_for_connection()'-call

Once you have an instance of TCPIPConnection, you can do the following:

    for data in myConnection:
        if data is not None:
            # ... do somethine
        else:
            # connection is closed, you can iterate over it anymore...

"""
class TCPIPConnection:
    def __init__(self, connection, remote_address, header_size=8, buffer_size=1024):
        self.connection = connection
        self.remote_address = remote_address
        self.header_size = header_size
        self.buffer_size = max(header_size, buffer_size)
        self.header = ''

    def is_open(self):
        return self.connection is not None

    def __iter__(self):
        return self

    def next(self):
        """
        Returns the next batch of data from the connection.
        NOTE: This does NOT need to be buffer_size, it can also be lower 
        than that. In case the low-level buffer-size is smaller than specified
        here, it will always be lower than that.
        It returns raw bytes, nothing else
        When the connection is closed, it returns None
        """
        if self.connection is not None:
            data = self.connection.recv(self.buffer_size)
            if data:
                return data
            else:
                self.connection.close()
                self.connection = None
        return None

    def get_header(self):
        return self.header

    def send_data(self, data):
        """
        Sends `data` through the open connection
        Returns (flag, num_bytes_sent)
        If it couldn't send all data, it will return False, num_bytes
        If the connection was already closed, it will return False, -1
        """
        if self.connection is not None:
            total_sent = 0
            total_to_send = len(data)
            if total_to_send > 0:
                tts_str = '{:08d}'.format(total_to_send)
                self.connection.send(tts_str)
            while len(data) > 0:
                sent_len = self.connection.send(data)
                total_sent += sent_len
                if sent_len != len(data):
                    data = data[sent_len:]
                else:
                    data = ''
            if total_sent == total_to_send:
                return True, total_sent
            else:
                return False, total_sent
        else:
            return False, -1

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        return True




"""
TCPIPListener

This is the only class that you instantiate yourself. After instantiating,
you send a call to 'wait_for_connection', which then will give you an instance
of TCPIPReceiver. That's all you have to do...
"""
class TCPIPListener:
    def __init__(self, ip_address, port, header_size=32, buffer_size=1024):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((ip_address, port))
        self.socket.listen(1)
        self.header_size = header_size
        self.buffer_size = buffer_size

    def wait_for_connection(self):
        connection, remote_address = self.socket.accept()
        return TCPIPConnection(connection, remote_address, self.header_size, self.buffer_size)

    def shutdown(self):
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket = None


"""
class TCPIPConnector (used by clients to send data to server)
"""
class TCPIPConnector:
    def __init__(self, remote_ip, remote_port, read_buffer_size=1024, write_buffer_size=1024):
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.read_buffer_size = read_buffer_size
        self.write_buffer_size = write_buffer_size

    def connect(self):
        connection_successfull = False
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            connection.connect((self.remote_ip, self.remote_port))
            connection_successfull = True
        except:
            print('Could not connect to remote socket, maybe server is not running (IP:{}, PORT:{})'.format(self.remote_ip, self.remote_port))
        if connection_successfull:
            tcp_client = TCPIPConnection(connection, self.remote_ip)
            return tcp_client
        return None


