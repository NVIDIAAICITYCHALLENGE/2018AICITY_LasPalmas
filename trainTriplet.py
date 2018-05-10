from keras.callbacks import ModelCheckpoint
from model import TripletNet

import h5py
import sys
import numpy as np
from random import randint


def getDB_size():
    h5 = h5py.File(sys.argv[1], 'r')
    Y = h5['Y_ID']
    l = len(Y)
    h5.close()
    return l

def gen2(batch_size):#(x_train, y_train, batch_size):
    h5 = h5py.File(sys.argv[1], 'r')

    while True:
        randIndex = randint(0, size_set-20000-1)

        x_train = h5['X'][randIndex:randIndex+20000]
        y_train = h5['Y_ID'][randIndex:randIndex+20000]

        classes = np.unique(y_train)
        indices = {c: np.where(y_train[:] == c)[0] for c in classes}

        orig_classes = np.random.choice(classes, (batch_size,))
        comp_classes = np.random.choice(classes, (batch_size,))
        while sum(orig_classes == comp_classes) > 0:
            comp_classes = np.random.choice(classes, (batch_size,))
        x1 = []
        x = []
        x2 = []
        y = []
        for i in range(batch_size):
            x_index = np.random.choice(indices[orig_classes[i]], 1,)
            x1_index = np.random.choice(indices[orig_classes[i]], 1,)
            x2_index = np.random.choice(indices[comp_classes[i]], 1,)
            x.append(x_train[x_index].astype('float32') / 255)
            x1.append(x_train[x1_index].astype('float32') / 255)
            x2.append(x_train[x2_index].astype('float32') / 255)
            y.append(1)

        yield ([np.array(x).reshape(batch_size, x_train.shape[1], x_train.shape[2], x_train.shape[3]),
            np.array(x1).reshape(batch_size,x_train.shape[1],x_train.shape[2],x_train.shape[3]),
            np.array(x2).reshape(batch_size,x_train.shape[1],x_train.shape[2],x_train.shape[3])],
            np.expand_dims(np.array(y),axis=1)) #el 1 era y


size_set = getDB_size()

input_size = (80, 80, 3)
embedding_dimensions = int(sys.argv[3])#128
batch_size = int(sys.argv[4])#64
gen = gen2(batch_size)#TripletGenerator()
genV = gen2(batch_size)
t = TripletNet(shape=input_size, dimensions=embedding_dimensions)

checkpointer = ModelCheckpoint(
    filepath=sys.argv[2]+'modelBatch'+str(batch_size)+'embe'+str(embedding_dimensions)+'.hdf5',
    verbose=1, save_best_only=True
)
steps_per_epoach = 20000 / batch_size
t.model.fit_generator(
    gen, int(steps_per_epoach), epochs=5, verbose=1,
    callbacks=[checkpointer], validation_data=genV, validation_steps=1
)