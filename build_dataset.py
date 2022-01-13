import numpy as np
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sidekick.io.hdf5_writer import Hdf5Writer
from imutils import paths
import cv2
import os
import progressbar
import json
import argparse

ap= argparse.ArgumentParser()
ap.add_argument('--model_training', '-m', required=True, help='Flag to determine which model is trained. Choose from "angle" and "length".')
ap.add_argument('--input_dir', '-i', required=True, help='Path to input dir for images')
ap.add_argument('--train_output_file', '-to', required=True, help='Path to train output file. Must not exist by default.')
ap.add_argument('--val_output_file', '-vo', required=True, help='Path to val output file. Must not exist by default.')
ap.add_argument('--label_file', '-l', required=True, help='Path to input training labels.')

args= vars(ap.parse_args())

model_flag= args['model_training']
data_path= args['input_dir']
hdf5_train= args['train_output_file']
hdf5_test= args['val_output_file']
label_file= args['label_file']

class_to_use= []
f= open(label_file, 'r')
label_dict= json.loads(f.read())


train_paths= list(paths.list_images(data_path))
train_labels= [label_dict[t.split(os.path.sep)[-1]] for t in train_paths]

if model_flag=='angle':
    le= LabelEncoder()
    train_labels= le.fit_transform(train_labels)
    print(le.classes_)
    print("Number of classes are: {}".format(len(le.classes_)))

train_paths, test_paths, train_labels, test_labels= train_test_split(train_paths,train_labels,
                                               test_size=0.2)

print(train_paths[10], train_labels[10], test_paths[10], test_labels[10])

files= [('train', train_paths, train_labels, hdf5_train),
        ('val', test_paths, test_labels, hdf5_test)]

for optype, paths, labels, output_path in files:

    dat_writer= Hdf5Writer((len(paths), 224, 224), output_path)

    # Initializing the progress bar display
    display=["Building Dataset: ", progressbar.Percentage(), " ",
             progressbar.Bar(), " ", progressbar.ETA()]

    # Start the progress bar
    progress= progressbar.ProgressBar(maxval=len(paths), widgets=display).start()

    # Iterate through each img path
    for (i, (p, l)) in enumerate(zip(paths,labels)):
        img= cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        img= img.astype('float') / 255.0


        dat_writer.add([img], [l])
        progress.update(i)

    # Finish the progress for one type
    progress.finish()
    dat_writer.close()