from tensorflow.keras.utils import to_categorical
import h5py
import numpy as np

class Hdf5DataGen:
    def __init__(self, dbPath, batchSize, classes, encode=True, aug=None, preprocessors=None):

        self.db= h5py.File(dbPath, 'r')
        self.batchSize= batchSize
        self.num_classes= classes
        self.encode= encode
        self.aug= aug
        self.preprocessors= preprocessors

        self.data_length= self.db['Images'].shape[0]

    def generator(self, counter=np.inf):
        start=0

        while start< counter:
            for i in np.arange(0, self.data_length, self.batchSize):
                data = self.db['Images'][i:i+self.batchSize]
                labels = self.db['Labels'][i:i + self.batchSize]

                if self.encode:
                    labels= to_categorical(labels, self.num_classes)

                if self.preprocessors is not None:
                    processed_data=[]

                    for d in data:
                        for p in self.preprocessors:
                            d= p.preprocess(d)

                        processed_data.append(d)

                    data= np.array(processed_data)

                if self.aug is not None:
                    # Notice the next to get next value from generator
                    data, labels= next(self.aug.flow(
                        data, labels, batch_size= self.batchSize
                    ))

                yield (data, labels)

            start+=1

    def close(self):
        self.db.close()