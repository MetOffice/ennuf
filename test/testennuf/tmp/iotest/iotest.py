#  (C) Crown Copyright, Met Office, 2023.
import numpy as np


def iotest():
    rng = np.random.default_rng(43)
    data = rng.random((3, 2, 5))

    data = np.arange(3 * 2 * 5, dtype=np.float32)
    data = data.reshape((3, 2, 5), order='F')
    data.T.tofile('itest.dat')
    data = np.fromfile('otest.dat', dtype=np.float32, count=3*2*5)
    data = data.reshape((3, 2, 5), order='F')
    print(data)


if __name__ == '__main__':
    iotest()
