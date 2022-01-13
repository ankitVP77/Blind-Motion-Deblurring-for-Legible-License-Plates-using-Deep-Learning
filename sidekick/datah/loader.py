import cv2
import numpy as np
import os

class Loader:
    def __init__(self,preprocessors=None):
        self.preprocessors=preprocessors
        if self.preprocessors is None:
            self.preprocessors=[]

    def load(self,imgpaths,verbose=-1):
        data=[]
        labels=[]

        for (i,imgpath) in enumerate(imgpaths):
            image=cv2.imread(imgpath)
            label= imgpath.split(os.path.sep)[-2]

            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image=p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i >0 and (i+1)% verbose==0:
                print("processed:{}/{}".format(i+1,len(imgpaths)))

        print("Done!")
        return (np.array(data),np.array(labels))
