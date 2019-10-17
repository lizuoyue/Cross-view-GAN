from PIL import Image
import glob
import numpy as np
import os
import tqdm

two_dim = np.array([
	[255, 255, 0],
	[  0, 127, 0],
	[127,   0, 0],
	[  0, 255, 0],
	[255,   0, 0],
])

target = '../dataset/L2R_Bicycle_No_Depth'
os.makedirs(target, exist_ok = True)
for mode in ['/test', '/train']:
	os.makedirs(target + mode.replace('test', 'val'), exist_ok = True)
	dataset_dir = '../dataset/L2R_Zuoyue' + mode
	path = dataset_dir + '/*_street_sem_label.png'
	files = glob.glob(path)
	for file in tqdm.tqdm(files):
		sem = np.array(Image.open(file))
		rgb = np.array(Image.open(file.replace('sem_label', 'rgb')))
		info = rgb.copy()
		for i in range(5):
			info[sem == i, :] = two_dim[i]
		bi = np.concatenate([info, rgb], 1)
		basename = '/' + os.path.basename(file)
		Image.fromarray(bi).save(target + mode.replace('test', 'val') + basename.replace('_street_sem_label', ''))
