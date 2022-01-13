import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse

ap= argparse.ArgumentParser()
ap.add_argument('--image', '-i', required=True, help='Path to input blurred image')
ap.add_argument('--angle_model', '-a', required=True, help='Path to trained angle model')
ap.add_argument('--length_model', '-l', required=True, help='Path to trained length model')
args= vars(ap.parse_args())

def process(ip_image, length, deblur_angle):
    noise = 0.01
    size = 200
    length= int(length)
    angle = (deblur_angle*np.pi) /180

    psf = np.ones((1, length), np.float32) #base image for psf
    costerm, sinterm = np.cos(angle), np.sin(angle)
    Ang = np.float32([[-costerm, sinterm, 0], [sinterm, costerm, 0]])
    size2 = size // 2
    Ang[:,2] = (size2, size2) - np.dot(Ang[:,:2], ((length-1)*0.5, 0))
    psf = cv2.warpAffine(psf, Ang, (size, size), flags=cv2.INTER_CUBIC) #Warp affine to get the desired psf
#     cv2.imshow("PSF",psf)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

    gray = ip_image
    gray = np.float32(gray) / 255.0
    gray_dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT of the image
    psf /= psf.sum() #Dividing by the sum
    psf_mat = np.zeros_like(gray)
    psf_mat[:size, :size] = psf
    psf_dft = cv2.dft(psf_mat, flags=cv2.DFT_COMPLEX_OUTPUT) #DFT of the psf
    PSFsq = (psf_dft**2).sum(-1)
    imgPSF = psf_dft / (PSFsq + noise)[...,np.newaxis] #H in the equation for wiener deconvolution
    gray_op = cv2.mulSpectrums(gray_dft, imgPSF, 0)
    gray_res = cv2.idft(gray_op,flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT) #Inverse DFT
    gray_res = np.roll(gray_res, -size//2,0)
    gray_res = np.roll(gray_res, -size//2,1)

    return gray_res


# Function to visualize the Fast Fourier Transform of the blurred images.
def create_fft(img):
    img = np.float32(img) / 255.0
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag_spec = 20 * np.log(np.abs(fshift))
    mag_spec = np.asarray(mag_spec, dtype=np.uint8)

    return mag_spec

# Change this variable with the name of the trained models.
angle_model_name= args['angle_model']
length_model_name= args['length_model']
model1= load_model(angle_model_name)
model2= load_model(length_model_name)

# read blurred image
ip_image = cv2.imread(args['image'])
ip_image=  cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
ip_image= cv2.resize(ip_image, (640, 480))
# FFT visualization of the blurred image
fft_img= create_fft(ip_image)

# Predicting the psf parameters of length and angle.
img= cv2.resize(create_fft(ip_image), (224,224))
img= np.expand_dims(img_to_array(img), axis=0)/ 255.0
preds= model1.predict(img)
# angle_value= np.sum(np.multiply(np.arange(0, 180), preds[0]))
angle_value = np.mean(np.argsort(preds[0])[-3:])

print("Predicted Blur Angle: ", angle_value)
length_value= model2.predict(img)[0][0]
print("Predicted Blur Length: ",length_value)

op_image = process(ip_image, length_value, angle_value)
op_image = (op_image*255).astype(np.uint8)
op_image = (255/(np.max(op_image)-np.min(op_image))) * (op_image-np.min(op_image))

cv2.imwrite("result.jpg", op_image)

