from PIL import Image
import glob
import numpy as np
import os
import tqdm

mode = 'train'
dataset_dir = '../dataset/L2R_Zuoyue/%s' % mode
os.makedirs(dataset_dir, exist_ok = True)

path = '../dataset/L2R/%s/*_street_sem2.png' % mode
files = glob.glob(path)
for file in tqdm.tqdm(files):
	os.popen('cp %s %s' % (file, dataset_dir))
	for item in ['_proj_depth.png', '_proj_rgb.png', '_street_rgb.png']:
		os.popen('cp %s %s' % (file.replace('_street_sem2.png', item), dataset_dir))
	os.popen('cp %s %s' % (file.replace('_street_sem2.png', '_sate_rgb.jpg').replace('L2R', 'R2D'), dataset_dir))

# path = '../dataset/R2D/train/*_sate_rgb.jpg' # 3393
# path = '../dataset/L2R/train/*_proj_depth.png' # 2935
# path = '../dataset/L2R/train/*_proj_rgb.png' # 2935
# path = '../dataset/L2R/train/*_street_sem2.png' # 2935
