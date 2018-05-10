__author__ = 'pedro'

import cv2
import h5py
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-db", "--database_path", help="ruta de la base de datos", nargs='?', type=str, default='/home/pedro/caffe/examples/nvidiaChallange/dbLoc1_1.hdf5')
parser.add_argument("-f", "--file_path", help="ruta al fichero de detecciones", type=str, nargs='?', default='/home/pedro/caffe/examples/nvidiaChallange/resultChallangeEmbe50thr2750.txt')
parser.add_argument("-out", "--out", help="ruta imagenes terminada en /", type=str, nargs='?', default='/home/pedro/caffe/examples/nvidiaChallange/prueba/')
parser.add_argument("-v", "--verbose", help="Modo depuracion", action="store_true")

args = parser.parse_args()
if args.verbose:
    print "depuracion activada!!!"

db = h5py.File(args.database_path, 'r')
x = db['X']
y = db['Y_ID']
desc = db['desc']
file = open(args.file_path, 'r')
out = args.out

index = 0
for linea in file:
    index += 1
    carsID = np.array(linea.strip().split(" ")[1:]).astype(int)
    indexs = [np.where(k == y[:]) for k in carsID]
    val = [1 for item in indexs if len(item[0]) > 1]
    if len(val) == 1:
        for index_per_ids in indexs:
            i = index_per_ids[0][0]
            cv2.imwrite(out + str(index) + '-' +str(i) + '.jpeg', x[i])
            pass