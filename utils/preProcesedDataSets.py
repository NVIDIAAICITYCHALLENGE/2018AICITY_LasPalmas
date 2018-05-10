__author__ = 'pedro'

import sys
import h5py
import tables
import numpy as np

def load_nvidiaVideo(percent = 0.7):
    h5 = h5py.File(db_path, 'r')
    X = h5['X']
    Y_id = np.expand_dims(h5['Y_ID'], axis=1)
    Y_desc = h5['desc']
    return (X[0:int(percent*len(X))], Y_id[0:int(percent*len(Y_id))], Y_desc[0:int(percent*len(Y_id))]),\
           (X[int(percent*len(X)):],Y_id[int(percent*len(Y_id)):], Y_desc[int(percent*len(Y_id)):])

def inizialize_dataset():
    h5 = tables.open_file(dbOut_path, mode='w')
    data_shape = (0, sizedb[0], sizedb[1], sizedb[2])
    img_dtype = tables.UInt8Atom()
    label_dtype = tables.UInt64Atom()
    X_storage = h5.create_earray(h5.root, 'X', img_dtype, shape=data_shape)
    Y_storageID = h5.create_earray(h5.root, 'Y_ID', label_dtype, shape=(0,))
    Y_desc = h5.create_earray(h5.root, 'desc', label_dtype, shape=(0,6))
    return X_storage, Y_storageID, Y_desc

def idInGhosts(id):
    ghostFile = open(ghostFile_path, 'r')
    for line in ghostFile:
        file_id = int(line.strip())
        if id == file_id:
            ghostFile.close()
            return True
    ghostFile.close()
    return False

db_path = sys.argv[1]
ghostFile_path = sys.argv[2]
dbOut_path = sys.argv[3]

sizedb = [80, 80, 3]

(x_train, id_train, desc_train), (x_test, id_test, desc_test) = load_nvidiaVideo(1)

new_X, new_YID, new_desc = inizialize_dataset()

for i in range(np.min(id_train), int(np.max(id_train)+1)):
    indexes = np.where(id_train == i)
    if len(indexes[0]) > 5 and not idInGhosts(i):
        for index in indexes[0]:
            new_X.append(x_train[index][None])
            new_YID.append(np.array([id_train[index][0]]))
            new_desc.append(desc_train[index][None])