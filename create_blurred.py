import numpy as np
import os
import cv2
import random
import json
import argparse

ap= argparse.ArgumentParser()
ap.add_argument('--input_dir', '-i', required=True, help='Path to input dir for images')
ap.add_argument('--output_dir', '-o', required=True, help='Path to output dir to store files. Must be created')
ap.add_argument('--max_imgs', '-m', default=20000, type=int, help='Max number of images to generate')

args= vars(ap.parse_args())

def apply_motion_blur(image, size, angle):
	k = np.zeros((size, size), dtype=np.float32)
	k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
	k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
	k = k * ( 1.0 / np.sum(k) )        
	return cv2.filter2D(image, -1, k) 


folder = args['input_dir']
folder_save = args['output_dir']
max_images = args['max_imgs']

print(max_images)


labels_angle = {}
labels_length= {}
images_done = 0
for filename in os.listdir(folder):
	img = cv2.imread(os.path.join(folder,filename))
	if img is not None and img.shape[1] > img.shape[0]:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_resized = cv2.resize(img_gray, (640,480), interpolation = cv2.INTER_AREA)
		length = random.randint(20,40)
		angle = random.randint(0,359)
		blurred = apply_motion_blur(img_resized, length, angle)
		cv2.imwrite(os.path.join(folder_save,filename), blurred)
		if angle>=180:
			angle_a= angle - 180
		else:
			angle_a= angle
		labels_angle[filename] = angle_a
		labels_length[filename]= length
		images_done += 1
		print("%s done"%images_done)
		if(images_done == max_images):
			print('Done!!!')
			break

with open('angle_labels.json', 'w') as file:
	json.dump(labels_angle, file)
with open('length_labels.json', 'w') as file:
	json.dump(labels_length, file)
    

