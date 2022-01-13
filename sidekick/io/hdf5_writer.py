import h5py
import os

class Hdf5Writer:
    def __init__(self, dims, outputPath, dbName='Images', buffSize= 1000):

        # throw an error if the file already exists
        if os.path.exists(outputPath):
            raise ValueError("PATH ALREADY PRESENT. PLEASE DELETE FILES"
                             "BEFORE PROCEEDING.")

        # database to store data
        self.db= h5py.File(outputPath, 'w')
        # define dataset containers to store data and labels
        self.data= self.db.create_dataset(dbName, dims, dtype='float')
        self.labels= self.db.create_dataset('Labels', shape=(dims[0],), dtype='int')

        # defining a buffer and index variable for the buffer
        self.buffSize= buffSize
        self.buffer= {"data": [], "labels": []}
        self.idx= 0

    def add(self, values, labels):
        self.buffer['data'].extend(values)
        self.buffer['labels'].extend(labels)

        if len(self.buffer['data'])>=self.buffSize:
            self.flush()

    def flush(self):
        # When buffer size is reached flush data to dataset container
        temp_idx= self.idx + len(self.buffer['data'])
        # index from prev_idx to new_idx
        self.data[self.idx:temp_idx]= self.buffer['data']
        self.labels[self.idx:temp_idx]= self.buffer['labels']
        # update new_idx
        self.idx=temp_idx
        # reinitialize the buffer
        self.buffer={'data': [], 'labels': []}

    def flushClassNames(self, classNames):

        # Creating a special
        labelNames= self.db.create_dataset('Label_Names', (len(classNames),),
                                           dtype=h5py.special_dtype(vlen=unicode))
        labelNames[:]= classNames

    def close(self):

        if len(self.buffer['data'])>0:
            self.flush()

        self.db.close()