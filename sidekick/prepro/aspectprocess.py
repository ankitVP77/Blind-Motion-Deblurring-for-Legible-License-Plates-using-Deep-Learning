import imutils
import cv2

class AspectProcess:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width= width
        self.height= height
        self.inter= inter

    def preprocess(self, image):
        h,w= image.shape[0:2]
        dh=0
        dw=0

        # Resize the smaller dimension and calculate the offset of the other dim
        # The offset is chosen such that a centre crop is formed
        if w<h:
            image= imutils.resize(image, width=self.width, inter=self.inter)
            dh= int((image.shape[0]- self.height)/2.0)
        else:
            image= imutils.resize(image, height= self.height, inter= self.inter)
            dw= int((image.shape[1]- self.width)/2.0)

        h,w= image.shape[0:2]
        # Center crop
        image= image[dh:h-dh, dw:w-dw]

        return cv2.resize(image, (self.width, self.height),interpolation=self.inter)