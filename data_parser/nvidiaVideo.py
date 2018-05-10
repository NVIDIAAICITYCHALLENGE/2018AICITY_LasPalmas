__author__ = 'pedro'

import h5py
import numpy as np
import shutil
import sys
import cv2
import dlib
import os
import tables

from nvidiaChallange.ssd_detector.tensor_sampling_utils import sample_tensors
from keras.optimizers import Adam
from keras import backend as K
from nvidiaChallange.ssd_detector.keras_ssd300 import ssd_300
from nvidiaChallange.ssd_detector.keras_ssd_loss import SSDLoss
from nvidiaChallange.ssd_detector.ssd_box_encode_decode_utils import decode_y

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
db_path = sys.argv[2]
idsGhostFile = open(sys.argv[3], 'a')
idContinueFilePath = sys.argv[4]
KILL_GHOST = int(sys.argv[5])
starting_offset = int(sys.argv[6])
img_height = int(sys.argv[7])
img_width = int(sys.argv[8])

weights_source_path = '/homes/pamarinreyes/VGG_coco_SSD_300x300_iter_400000.h5'
weights_destination_path = '/homes/pamarinreyes/VGG_coco_SSD-2_300x300_iter_400000.h5'
sizedb = [80, 80, 3]
img_channels = 3 # Number of color channels of the input images
subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = False # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# TODO: Set the number of classes.
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = True # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

def encode_videoName(video_path):
    if video_path.rfind('1_1') != -1:
       return 1
    if video_path.rfind('1_2') != -1:
       return 2
    if video_path.rfind('1_3') != -1:
       return 3
    if video_path.rfind('1_4') != -1:
       return 4
    if video_path.rfind('2_1') != -1:
       return 5
    if video_path.rfind('2_2') != -1:
       return 6
    if video_path.rfind('2_3') != -1:
       return 7
    if video_path.rfind('2_4') != -1:
       return 8
    if video_path.rfind('2_5') != -1:
       return 9
    if video_path.rfind('2_6') != -1:
       return 10
    if video_path.rfind('3_1') != -1:
       return 11
    if video_path.rfind('3_2') != -1:
       return 12
    if video_path.rfind('4_1') != -1:
       return 13
    if video_path.rfind('4_2') != -1:
       return 14
    if video_path.rfind('4_3') != -1:
       return 15
    return -1

def inizialize_dataset():
    global X_storage, Y_storageID, desc_storage
    h5 = tables.open_file(db_path, mode='w')
    data_shape = (0, sizedb[0], sizedb[1], sizedb[2])
    img_dtype = tables.UInt8Atom()
    label_dtype = tables.UInt64Atom()
    X_storage = h5.create_earray(h5.root, 'X', img_dtype, shape=data_shape)
    Y_storageID = h5.create_earray(h5.root, 'Y_ID', label_dtype, shape=(0,))
    desc_storage = h5.create_earray(h5.root, 'desc', label_dtype, shape=(0,6)) #video,frame,boundingbox


def detectMultipleVehicles():
    y_pred = model.predict(img[np.newaxis, :, :, :])
    y_pred_decoded = decode_y(y_pred,
                              confidence_thresh=0.80,
                              iou_threshold=0.9,
                              top_k=200,
                              input_coords='centroids',
                              normalize_coords=normalize_coords,
                              img_height=img_height,
                              img_width=img_width)
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_decoded[0])
    return y_pred_decoded

def getBoundingBox(entity):
    xmintrac = entity[3].get_position().left()
    ymintrac = entity[3].get_position().top()
    xmaxtrac = entity[3].get_position().right()
    ymaxtrac = entity[3].get_position().bottom()
    return xmaxtrac, xmintrac, ymaxtrac, ymintrac

def fuckGhosts(entities_list):
    for entity in entities_list:
        if entity[len(entity)-1] >= KILL_GHOST:
            idsGhostFile.write(str(entity[0]) + '\n')
            entities_list.remove(entity)

def updateTrackers(entities_list):
    for entity in entities_list:
        idd = entity[0]
        entity[3].update(img)
        xmaxtrac, xmintrac, ymaxtrac, ymintrac = getBoundingBox(entity)
        entity[1] = (xmintrac + xmaxtrac) / 2
        entity[2] = (ymintrac + ymaxtrac) / 2
        entity[len(entity)-1] += 1

        saveSample(img, encode_videoName(video_path), nframes, ymintrac, ymaxtrac, xmintrac, xmaxtrac, idd)
        #cv2.rectangle(img, (int(xmintrac), int(ymintrac)), (int(xmaxtrac), int(ymaxtrac)), (255,0,0), 2)
        #cv2.putText(img, str(idd), (int(xmintrac), int(ymintrac)), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)

def removeTrackers(entities_list):
    for entity in entities_list:
        xmaxtrac, xmintrac, ymaxtrac, ymintrac = getBoundingBox(entity)
        condLimit = ymintrac <= 0 or ymaxtrac >= img_height or xmintrac <= 0 or xmaxtrac >= img_width
        if condLimit:
            entities_list.remove(entity)

def indexOfBox(lista, box):
    xmin = box[2]
    ymin = box[3]
    xmax = box[4]
    ymax = box[5]
    for i in range(0, len(lista)):
        if xmin < lista[i][1] and lista[i][1] < xmax and ymin < lista[i][2] and lista[i][2] < ymax:
            return i
    return -1

def generateFineTuningDetectorModel():
    global model
    # Make a copy of the weights file.
    shutil.copy(weights_source_path, weights_destination_path)
    # Load both the source weights file and the copy we made.
    # We will load the original weights file in read-only mode so that we can't mess up anything.
    weights_source_file = h5py.File(weights_source_path, 'r')
    weights_destination_file = h5py.File(weights_destination_path)
    classifier_names = ['conv4_3_norm_mbox_conf',
                        'fc7_mbox_conf',
                        'conv6_2_mbox_conf',
                        'conv7_2_mbox_conf',
                        'conv8_2_mbox_conf',
                        'conv9_2_mbox_conf']
    conv4_3_norm_mbox_conf_kernel = weights_source_file[classifier_names[0]][classifier_names[0]]['kernel:0']
    conv4_3_norm_mbox_conf_bias = weights_source_file[classifier_names[0]][classifier_names[0]]['bias:0']

    # TODO: Set the number of classes in the source weights file. Note that this number must include
    #       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
    n_classes_source = 81
    # TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
    #       In case you would like to just randomly sample a certain number of classes, you can just set
    #       `classes_of_interest` to an integer instead of the list below.
    classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]
    n_classes = len(classes_of_interest) - 1
    for name in classifier_names:
        # Get the trained weights for this layer from the source HDF5 weights file.
        kernel = weights_source_file[name][name]['kernel:0'].value
        bias = weights_source_file[name][name]['bias:0'].value

        # Get the shape of the kernel. We're interested in sub-sampling
        # the last dimension, 'o'.
        height, width, in_channels, out_channels = kernel.shape

        # Compute the indices of the elements we want to sub-sample.
        # Keep in mind that each classification predictor layer predicts multiple
        # bounding boxes for every spatial location, so we want to sub-sample
        # the relevant classes for each of these boxes.
        subsampling_indices = []
        for i in range(int(out_channels / n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))

        # Sub-sample the kernel and bias.
        # The `sample_tensors()` function used below provides extensive
        # documentation, so don't hesitate to read it if you want to know
        # what exactly is going on here.
        new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                              sampling_instructions=[height, width, in_channels, subsampling_indices],
                                              axes=[[3]],
                                              # The one bias dimension corresponds to the last kernel dimension.
                                              init=['gaussian', 'zeros'],
                                              mean=0.0,
                                              stddev=0.005)

        # Delete the old weights from the destination file.
        del weights_destination_file[name][name]['kernel:0']
        del weights_destination_file[name][name]['bias:0']
        # Create new datasets for the sub-sampled weights.
        weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
        weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)
    conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
    conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']
    #print("Shape of the '{}' weights:".format(classifier_names[0]))
    #print()
    #print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
    #print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

    K.clear_session()  # Clear previous models from memory.
    model = ssd_300(image_size=(img_height, img_width, img_channels),
                    n_classes=n_classes,
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_per_layer=aspect_ratios,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    limit_boxes=limit_boxes,
                    variances=variances,
                    coords=coords,
                    normalize_coords=normalize_coords,
                    subtract_mean=subtract_mean,
                    divide_by_stddev=None,
                    swap_channels=swap_channels)

    weights_path = weights_destination_path
    model.load_weights(weights_path, by_name=True)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

def saveSample(im, video, nframe, top, bottom, left, right, idd):
    if abs(right-left) > sizedb[0] and top >=0 and bottom <= im.shape[0] and left >= 0 and right <= im.shape[1]:
        crop_img = im[int(top):int(bottom), int(left):int(right)]
        X_storage.append(cv2.resize(crop_img,
                                            (sizedb[0], sizedb[1]), interpolation=cv2.INTER_CUBIC).reshape(sizedb[0], sizedb[1],  sizedb[2])[None])
        Y_storageID.append(np.array([idd]))
        desc_storage.append(np.array([video, nframe, top, bottom, left, right])[None])


inizialize_dataset()
generateFineTuningDetectorModel()

ret = True
nframes = 0
entities_list = []

if os.path.isfile(idContinueFilePath):
    idContinueFile = open(idContinueFilePath, 'r')
    lastid = int(idContinueFile.readlines()[-1].strip())
    idContinueFile.close()
else:
    lastid = 0
idContinuedFile = open(idContinueFilePath, 'a')
n_diff_detect = lastid

while True:
    ret, frame = cap.read()
    nframes += 1
    if ret == False:
        break
    img = frame
    if nframes >= 0: # to process a from a specific point of a video
        y_pred_decoded = detectMultipleVehicles()

        fuckGhosts(entities_list)
        removeTrackers(entities_list)
        updateTrackers(entities_list)

        for box in y_pred_decoded[0]:
            class_id = box[0]
            confidence = box[1]
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]

            pos = indexOfBox(entities_list, box)
            if class_id == 7. or class_id == 2. or not (int((ymax+ymin)/2) < img_height - starting_offset and int((ymax+ymin)/2) > starting_offset and
                                                                int((xmax+xmin)/2) > starting_offset and int((xmax+xmin)/2) < img_width - starting_offset): #whole car validation
                continue

            if pos == -1:
                n_diff_detect += 1
                idd = n_diff_detect
                xm = (xmin + xmax) / 2
                ym = (ymin + ymax) / 2
                v = [idd, xm, ym, dlib.correlation_tracker(), class_id, 1] # the last value represents the number of frames
                v[3].start_track(img, dlib.rectangle(int(xmin), int(ymin), int(xmax), int(ymax)))
                entities_list.append(v)

                saveSample(img, encode_videoName(video_path), nframes, ymin, ymax, xmin, xmax, idd)
                #cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,0,0), 2)
                #cv2.putText(img, str(idd), (int(xmin), int(ymin)), cv2.FONT_HERSHEY_PLAIN, 1,(255,255,255), 2)

    #cv2.imwrite(folder + str(nframes).zfill(10) +'.jpeg', img)

cap.release()
idsGhostFile.flush()
idsGhostFile.close()