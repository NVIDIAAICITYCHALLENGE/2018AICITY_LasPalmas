__author__ = 'pedro'

import sys
import numpy as np
import tables
import h5py
from model import TripletNet


def inizialize_dataset():
    h5 = tables.open_file(dbOut_path, mode='w')
    data_shape = (0, size_embedding)
    img_dtype = tables.Float32Atom()
    label_dtype = tables.UInt64Atom()
    X_storage = h5.create_earray(h5.root, 'X', img_dtype, shape=data_shape)
    Y_storageID = h5.create_earray(h5.root, 'Y_ID', label_dtype, shape=(0,))
    Y_desc = h5.create_earray(h5.root, 'desc', label_dtype, shape=(0,6))
    return X_storage, Y_storageID, Y_desc


db_path = sys.argv[1]
dbOut_path = sys.argv[2]
model_path = sys.argv[3]
size_embedding = int(sys.argv[4])

sizedb = [80, 80, 3]

h5 = h5py.File(db_path, 'r')
x_train = h5['X']
id_train = h5['Y_ID']
desc_train = h5['desc']

new_X, new_YID, new_desc = inizialize_dataset()
prev_id = id_train[0]
iterations = 0

t = TripletNet(shape=sizedb, dimensions=size_embedding, train=False)
t.model.load_weights(model_path, by_name=False)
borrar=0
for index in range(len(id_train)):
    curr_id = id_train[index]
    if curr_id != prev_id:
        borrar+=1
        index_sel = (index -1)- int(iterations/2)
        embeding = t.model.predict([x_train[index_sel][np.newaxis,:,:,:], x_train[index_sel][np.newaxis,:,:,:],
                                    x_train[index_sel][np.newaxis,:,:,:]])[1][0]
        new_X.append(embeding[None])
        new_YID.append(np.array([id_train[index_sel]]))
        new_desc.append(desc_train[index_sel][None])

        iterations = 0
        prev_id = curr_id

    iterations += 1