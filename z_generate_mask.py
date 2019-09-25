from PIL import Image
import glob
import numpy as np
import os
import tqdm

colors = [
	[ 70, 130, 180], # sky
	[ 70,  70,  70], # building
	[128,  64, 127], # road
	[106, 142,  34], # vegetation
	[243,  36, 232], # sidewalk
]

mode = 'train'
dataset_dir = '../dataset/L2R_Zuoyue/%s' % mode
files = glob.glob(dataset_dir + '/*_street_sem2.png')

for file in tqdm.tqdm(files):
	img = np.array(Image.open(file)).astype(np.int32)
	dist = []
	for color in colors:
		rgb = np.array(color)
		dist.append(np.sum((img - rgb) ** 2, axis = -1))
	dist = np.array(dist)
	label = dist.argmin(axis = 0).astype(np.uint8)
	Image.fromarray(label).save(file.replace('street_sem2', 'street_sem_label'))
