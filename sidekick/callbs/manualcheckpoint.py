from tensorflow.keras.callbacks import Callback
import os

class ManualCheckpoint(Callback):
    def __init__(self, output, save_at=3, start_from=0):
        super(Callback, self).__init__()

        self.output= output
        self.save_at= save_at
        self.initial_epoch= start_from

    def on_epoch_end(self, epoch, logs={}):
        if (self.initial_epoch+1) % self.save_at==0:
            save_path= os.path.sep.join([self.output,
                                         "weights-epoch {}.hdf5".format(self.initial_epoch+1)])

            self.model.save(save_path, overwrite=True)

        self.initial_epoch+=1
