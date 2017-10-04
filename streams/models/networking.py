"""
A script for communicating to a remote machine that has Caffe installed so that model features could be returned for images provided by the local machine.
"""
from __future__ import absolute_import, division, print_function

import sys, traceback
import socket, subprocess, struct, shlex
import pickle

import numpy as np


class Socket(object):

    DTYPES = ['str', 'numpy.ndarray', 'object']
    ACTIONS = ['exec', 'call', 'error', 'hasattr']

    def __init__(self, host, port=None):
        self.host = host
        self.port = port
        self.islocal = True if port is None else False

    def set_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def encode(self, data, dtype=None, action='exec'):
        """
        Message format:
          - (4 bytes) meta parameter length
          - (4 bytes) data length
          - (4 bytes) data type
          - (4 bytes) action
          - (unlimited) meta parameters
          - (unlimited) data
        """
        if dtype is None:
            if isinstance(data, str):
                dtype = 'str'
            elif isinstance(data, np.ndarray):
                dtype = 'numpy.ndarray'
            else:
                dtype = 'object'

        meta = ''
        if dtype == 'numpy.ndarray':
            meta = {'shape': data.shape, 'dtype': str(data.dtype)}
            data = data.tostring()
        elif dtype == 'object':
            data = pickle.dumps(data, -1)

        dtype = self.DTYPES.index(dtype)
        action = self.ACTIONS.index(action)
        meta = str(meta)
        msg = struct.pack('>IQII', len(meta), len(data), dtype, action) + meta + data
        return msg

    def decode(self, msg):
        metalen, datalen, dtype, action = struct.unpack('>IQII', msg[:20])
        dtype = self.DTYPES[dtype]
        action = self.ACTIONS[action]

        meta = msg[20: 20 + metalen]
        data = msg[20 + metalen: 20 + metalen + datalen]
        return self._decode(data, meta, dtype)

    def _decode(self, data, meta, dtype):
        if len(meta) > 0:
            meta = eval(meta)

        if dtype == 'str':
            data = data
        elif dtype == 'numpy.ndarray':
            data = np.fromstring(data, dtype=meta['dtype'])
            data = data.reshape(meta['shape'])
        elif dtype == 'object':
            data = pickle.loads(data)
        else:
            data = None

        return data

    def send(self, data, dtype=None, action='exec'):
        msg = self.encode(data, dtype=dtype, action=action)
        self.conn.sendall(msg)

    def recv(self):
        header = self.recvall(20)
        if not header: return None

        metalen, datalen, dtype, action = struct.unpack('>IQII', header)
        dtype = self.DTYPES[dtype]
        action = self.ACTIONS[action]

        meta = self.recvall(metalen)
        data = self.recvall(datalen)
        data = self._decode(data, meta, dtype)

        if action == 'hasattr':
            self.recv_hasattr(data)
        elif data is not None and action == 'call':
            self.recv_func()
        elif action == 'error':
            print(data)
            sys.exit()
        else:
            return data

    def recvall(self, n):
        """
        Helper function to recv n bytes or return None if EOF is hit
        """
        data = ''
        while len(data) < n:
            packet = self.conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

    def listen(self):
        self.socket.listen(0)
        print('\r{}:{} -'.format(self.host, self.port), end='')
        sys.stdout.flush()
        self.conn, addr = self.socket.accept()
        print('\r{}:{} +'.format(self.host, self.port), end='')
        sys.stdout.flush()

    def send_hasattr(self, attr):
        """
        Requests if attribute is defined, and returns the answer
        """
        self.set_socket()
        self.conn = self.socket
        self.conn.connect((self.host, self.port))
        self.send(attr, action='hasattr')

        data = None
        while data is None:
            data = self.recv()

        self.conn.close()
        return data

    def recv_hasattr(self, attr):
        """
        Sends back the answer if attribute is defined
        """
        try:
            getattr(self, attr)
        except:
            self.send(False)
        else:
            self.send(True)
        self.conn.close()
        self.listen()

    def send_func(self, *args, **kwargs):
        self.set_socket()
        self.conn = self.socket
        self.conn.connect((self.host, self.port))

        self.send('', action='call')
        self.send(self._func_name)
        eargs = [self.encode(arg) for arg in args]
        self.send(eargs)
        ekwargs = {key:self.encode(val) for key, val in kwargs.items()}
        self.send(ekwargs)

        data = None
        while data is None:
            data = self.recv()

        self.conn.close()
        return data

    def recv_func(self):
        func = self.recv()
        func = getattr(self, func)
        eargs = self.recv()
        args = [self.decode(earg) for earg in eargs]
        ekwargs = self.recv()
        kwargs = {key:self.decode(val) for key, val in ekwargs.items()}

        try:
            resp = func(*args, **kwargs)
        except:
            self.send(traceback.format_exc(sys.exc_info()), action='error')
        else:
            if resp is None: resp = 'ok'
            self.send(resp)

        self.conn.close()
        self.listen()


class Client(Socket):

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            if attr != 'hasattr':
                if self.hasattr(attr):
                    self._func_name = attr
                    return self.send_func
                else:
                    raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))
            else:
                return self.send_hasattr

    def start_server(self):
        # subprocess.check_call(['scp', os.path.realpath(__file__), self.host + ':mcluster.py'])
        cmd = ('ssh {} mcluster.py server -p {}').format(self.host, self.port)
        self.server = subprocess.Popen(shlex.split(cmd))


class Server(Socket):

    def __init__(self, host, port=None):
        super(Server, self).__init__(host, port=port)
        self.set_socket()
        self.socket.bind((self.host, self.port))
        self.listen()
        self.loop()

    def loop(self):
        while True: self.recv()