from PIL import Image
import glob
import numpy as np
import os
import tqdm

target = '../dataset/L2R_Bicycle'
os.makedirs(target, exist_ok = True)
for mode in ['/train', '/test']:
	os.makedirs(target + mode, exist_ok = True)
	dataset_dir = '../dataset/L2R_Zuoyue' + mode
	path = dataset_dir + '/*_street_sem2.png'
	files = glob.glob(path)
	for file in tqdm.tqdm(files):
		sem = np.array(Image.open(file))
		rgb = np.array(Image.open(file.replace('sem2', 'rgb')))
		bi = np.concatenate([sem, rgb], 1)
		basename = '/' + os.path.basename(file)
		Image.fromarray(bi).save(target + mode + basename.replace('_street_sem2', ''))
