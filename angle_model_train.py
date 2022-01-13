from sidekick.nn.conv.angle_model import MiniVgg
from sidekick.io.hdf5datagen import Hdf5DataGen
from sidekick.callbs.manualcheckpoint import ManualCheckpoint
from sidekick.callbs.trainmonitor import TrainMonitor
from sidekick.prepro.process import Process
from sidekick.prepro.imgtoarrayprepro import ImgtoArrPrePro
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import argparse

ap= argparse.ArgumentParser()
ap.add_argument('-o','--output', type=str, required=True ,help="Path to output directory to store metrics")
ap.add_argument('-m', '--model', help='Path to checkpointed model')
ap.add_argument('-e','--epoch', type=int, default=0, help="Starting epoch of training")
args= vars(ap.parse_args())

hdf5_train_path= "train.hdf5"
hdf5_val_path= "val.hdf5"
epochs= 50
lr= 1e-2
batch_size= 32
num_classes= 180
fig_path= args['output']+"train_plot.jpg"
json_path= args['output']+"train_values.json"

print('[NOTE]:- Building Dataset...\n')
pro= Process(224, 224)
i2a= ImgtoArrPrePro()

train_gen= Hdf5DataGen(hdf5_train_path, batch_size, num_classes, preprocessors=[pro, i2a])
val_gen= Hdf5DataGen(hdf5_val_path, batch_size, num_classes, preprocessors=[pro, i2a])


if args['model'] is None:
    print("[NOTE]:- Building model from scratch...")
    model= MiniVgg.build(224, 224, 1, num_classes)
    opt= SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
else:
    print("[NOTE]:- Building model {}\n".format(args['model']))
    model= load_model(args['model'])

callbacks= [ManualCheckpoint(args['output'], save_at=1, start_from=args['epoch']),
            TrainMonitor(figPath= fig_path, jsonPath= json_path, startAt=args['epoch'])]

print("[NOTE]:- Training model...\n")

model.fit_generator(train_gen.generator(),
                    steps_per_epoch=train_gen.data_length//batch_size,
                    validation_data= val_gen.generator(),
                    validation_steps= val_gen.data_length//batch_size,
                    epochs=epochs,
                    max_queue_size=10,
                    callbacks= callbacks,
                    initial_epoch=args['epoch'])
