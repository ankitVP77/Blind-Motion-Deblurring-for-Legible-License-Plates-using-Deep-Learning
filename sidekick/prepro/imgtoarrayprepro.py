from tensorflow.keras.preprocessing.image import img_to_array

class ImgtoArrPrePro:
    def __init__(self,dataFormat=None):
        self.dataFormat=dataFormat

    def preprocess(self,img):

        return img_to_array(img,data_format=self.dataFormat)