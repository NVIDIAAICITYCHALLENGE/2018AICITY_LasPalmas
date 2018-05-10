__author__ = 'pedro'

import h5py
import numpy as np
import shutil
import sys

from tensor_sampling_utils import sample_tensors

weights_source_path = '/home/pedro/politicos/VGG_coco_SSD_300x300_iter_400000.h5'
weights_destination_path = '/home/pedro/VGG_coco_SSD_300x300_iter_400000.h5'
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

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

n_classes_source = 81
classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

subsampling_indices = []
for i in range(int(324/n_classes_source)):
    indices = np.array(classes_of_interest) + i * n_classes_source
    subsampling_indices.append(indices)
subsampling_indices = list(np.concatenate(subsampling_indices))

print(subsampling_indices)

# TODO: Set the number of classes in the source weights file. Note that this number must include
#       the background class, so for MS COCO's 80 classes, this must be 80 + 1 = 81.
n_classes_source = 81
# TODO: Set the indices of the classes that you want to pick for the sub-sampled weight tensors.
#       In case you would like to just randomly sample a certain number of classes, you can just set
#       `classes_of_interest` to an integer instead of the list below.
classes_of_interest = [0, 3, 8, 1, 2, 10, 4, 6, 12]

# classes_of_interest = 9 # Uncomment this in case you want to just randomly sub-sample the last axis instead of providing a list of indices.

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
    for i in range(int(out_channels/n_classes_source)):
        indices = np.array(classes_of_interest) + i * n_classes_source
        subsampling_indices.append(indices)
    subsampling_indices = list(np.concatenate(subsampling_indices))

    # Sub-sample the kernel and bias.
    # The `sample_tensors()` function used below provides extensive
    # documentation, so don't hesitate to read it if you want to know
    # what exactly is going on here.
    new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                          sampling_instructions=[height, width, in_channels, subsampling_indices],
                                          axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
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

print("Shape of the '{}' weights:".format(classifier_names[0]))
print()
print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)

from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model

from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images
subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
swap_channels = True # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.
# TODO: Set the number of classes.
n_classes = 8 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO

#scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
#LO COMENTE PEDRO scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
scales = [0.075,0.1, 0.125, 0.15, 0.175, 0.2, 0.25] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids', 'corners', or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.
#img_height,img_width,img_channels esto estaba abajo
model = ssd_300(image_size=(1080, 1920, img_channels),
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

print("Model built.")

# 2: Load the sub-sampled weights into the model.

# Load the weights that we've just created via sub-sampling.
weights_path = weights_destination_path

model.load_weights(weights_path, by_name=True)

print("Weights file loaded:", weights_path)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'ymin', 'xmax', 'ymax'])

# TODO: Set the paths to your dataset here.
images_path = sys.argv[1] #'/home/pedro/QuartilImagesTest'
labels_path = sys.argv[2] #'/home/pedro/priorGroundTrue.csv'
batch_size=int(sys.argv[3])

dataset.parse_csv(images_dir=images_path,
                  labels_filename=labels_path,
                  input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'], # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.
                  include_classes='all',
                  random_sample=False)

print("Number of images in the dataset:", dataset.get_n_samples())


generator = dataset.generate(batch_size=batch_size,
                             shuffle=False, #estaba a True
                             train=False,
                             returns={'processed_labels',
                                      'filenames'},
                             equalize=False,
                             brightness=False,
                             flip=False,
                             translate=False,
                             scale=False,
                             max_crop_and_resize=False, #(300,300,0,3),
                             random_pad_and_resize=False,
                             random_crop=False,
                             crop=False,
                             resize=False,
                             gray=False,
                             limit_boxes=True,
                             include_thresh=0.40)

# Generate samples
max_ite = dataset.get_n_samples() / batch_size
ite = 0
while ite < max_ite:
    ite += 1
    r = next(generator)
    y_pred = model.predict(r[0])
    for myindex in range(0, len(r[0])):
        i = 0
        y_pred_decoded = decode_y(y_pred,
                                  confidence_thresh=0.95,
                                  iou_threshold=0.4,
                                  top_k=200,
                                  input_coords='centroids',
                                  normalize_coords=normalize_coords,
                                  img_height=1080,#img_height,
                                  img_width=1920)#img_width)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded[myindex])


        # Visualize the predictions.

        from matplotlib import pyplot as plt
        plt.switch_backend('TKagg')
        plt.figure(figsize=(20,12))
        #plt.imshow(r[0][i])
        plt.imshow(r[0][myindex]) #lo puse yo pedro

        current_axis = plt.gca()

        classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist',
                   'traffic_light', 'motorcycle', 'bus', 'stop_sign'] # Just so we can print class names onto the image instead of IDs
        # Draw the predicted boxes in blue
        for box in y_pred_decoded[myindex]:
            class_id = box[0]
            confidence = box[1]
            xmin = box[2] 
            ymin = box[3] 
            xmax = box[4] 
            ymax = box[5]
            label = '{}: {:.2f}'.format(classes[int(class_id)], confidence)
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
        #plt.savefig('/home/pedro/testNvidia2/prueba'+r[2][myindex][r[2][myindex].rfind('/')+1:r[2][myindex].find('.')]+'.jpeg')
	plt.savefig(sys.argv[4]+r[2][myindex][r[2][myindex].rfind('/')+1:r[2][myindex].find('.')]+'.jpeg')

        # Draw the ground truth boxes in green (omit the label for more clarity)
        #for box in r[1][i]:
        #    class_id = box[0]
        #    xmin = box[1]
        #    ymin = box[2]
        #    xmax = box[3]
        #    ymax = box[4]
        #    label = '{}'.format(classes[int(class_id)])
        #    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        #    #current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
