__author__ = 'pedro'
import sys
import os
import tables
import numpy as np
import h5py

pathTosets = sys.argv[1]

dsOut = sys.argv[2]
sizedb = [80, 80, 3]

def inizialize_dataset():
    tab = tables.open_file(dsOut, mode='w')
    data_shape = (0, sizedb[0], sizedb[1], sizedb[2])
    img_dtype = tables.UInt8Atom()
    label_dtype = tables.UInt64Atom()
    X_storage = tab.create_earray(tab.root, 'X', img_dtype, shape=data_shape)
    Y_storageID = tab.create_earray(tab.root, 'Y_ID', label_dtype, shape=(0,))
    Y_desc = tab.create_earray(tab.root, 'desc', label_dtype, shape=(0,6))
    return X_storage, Y_storageID, Y_desc

increment = 0
[new_X, new_YID, new_YDesc] = inizialize_dataset()
for [root, folder, files] in os.walk(pathTosets):
    for file in files:
        if file.find('.hdf5') != -1:
            h5 = h5py.File(pathTosets+'/'+file, 'r')
            X = h5['X']
            Y_id = np.expand_dims(h5['Y_ID'], axis=1)
            desc = h5['desc']
            if len(new_YID) == 0:
                increment = 0
            else:
                increment = np.max(new_YID)

            for index in range(0, len(Y_id)):
                new_X.append(X[index][None])
                new_YID.append(np.array([int(Y_id[index][0] + increment)]))
                new_YDesc.append(desc[index][None])
    break