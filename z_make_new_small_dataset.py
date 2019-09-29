from PIL import Image
import glob
import numpy as np
import os
import tqdm
import random
random.seed(7)

mode = 'train'
dataset_dir = '../dataset/L2R_Zuoyue_Small/%s' % mode
os.makedirs(dataset_dir, exist_ok = True)

path = '../dataset/L2R_Zuoyue/%s/*_street_sem2.png' % mode
files = glob.glob(path)
random.shuffle(files)
for file in tqdm.tqdm(files[:64]):
	os.popen('cp %s %s' % (file.replace('_street_sem2.png', '*'), dataset_dir))
