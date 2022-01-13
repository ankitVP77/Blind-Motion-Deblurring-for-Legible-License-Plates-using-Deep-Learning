import cv2
import numpy as np

class MeanProcess:
    def __init__(self, R_Mean, G_Mean, B_Mean):
        self.R_Mean= R_Mean
        self.G_Mean= G_Mean
        self.B_Mean= B_Mean

    def preprocess(self, image):
        image= np.float32(image)
        B,G,R= cv2.split(image)

        B-= self.B_Mean
        G-= self.G_Mean
        R-= self.R_Mean

        return cv2.merge([B, G, R])