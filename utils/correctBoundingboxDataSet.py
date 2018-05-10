__author__ = 'pedro'

import sys
import numpy as np
import tables
import h5py

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


db_path = sys.argv[1]
dbOut_path = sys.argv[2]
offsetX = int(sys.argv[3])
offsetY = int(sys.argv[4])


sizedb = [80, 80, 3]

(x_train, id_train, desc_train), (x_test, id_test, desc_test) = load_nvidiaVideo(1)

new_X, new_YID, new_desc = inizialize_dataset()

for index in range(len(id_train)):
    new_X.append(x_train[index][None])
    new_YID.append(np.array([id_train[index][0]]))
    b1=desc_train[index][2] + offsetY
    b2=desc_train[index][3] + offsetY
    b3=desc_train[index][4] + offsetX
    b4=desc_train[index][5] + offsetX
    v = [desc_train[index][0], desc_train[index][1], int(b1), int(b2), int(b3), int(b4)]
    new_desc.append(np.array(v)[None])