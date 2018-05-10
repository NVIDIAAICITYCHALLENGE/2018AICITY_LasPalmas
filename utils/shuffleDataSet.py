import numpy as np
import sys
import h5py

h5 = h5py.File(sys.argv[1], 'r+')
X = h5['X']
Y_id = h5['Y_ID']
desc = h5['desc']

np.random.seed(8)
np.random.shuffle(X)
np.random.seed(8)
np.random.shuffle(Y_id)
np.random.sheed(8)
np.random.shuffle(desc)
