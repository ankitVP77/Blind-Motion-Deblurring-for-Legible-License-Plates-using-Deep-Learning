from tensorflow.keras import callbacks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

class TrainMonitor(callbacks.BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainMonitor, self).__init__()

        self.figPath= figPath
        self.jsonPath= jsonPath
        self.startAt= startAt

    def on_train_begin(self, logs={}):
        self.H={}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        for keys, values in logs.items():
            l= self.H.get(keys, [])
            l.append(float(values))
            self.H[keys] = l

        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as f:
                f.write(json.dumps(self.H))
                f.close()

        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]), 1)
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_acc")
            plt.plot(N, self.H["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.savefig(self.figPath)
            plt.close()