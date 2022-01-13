import numpy as np
import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--input_dir', '-i', required=True, help='Path to input dir for images')
ap.add_argument('--output_dir', '-o', required=True, help='Path to output dir to store files. Must be created')

args= vars(ap.parse_args())


folder = args['input_dir']
folder_save = args['output_dir']


labels = {}
images_done = 0
for filename in os.listdir(folder):
	img = cv2.imread(os.path.join(folder,filename))
	if img is not None:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray = np.float32(img_gray) / 255.0
		f = np.fft.fft2(img_gray)
		fshift = np.fft.fftshift(f)
		mag_spec = 20 * np.log(np.abs(fshift))
		mag_spec = np.asarray(mag_spec, dtype=np.uint8)
		cv2.imwrite(os.path.join(folder_save,filename), mag_spec)
		images_done += 1
		print("%s done"%images_done)

