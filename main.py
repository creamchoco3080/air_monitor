import pandas as pd
import numpy as np
import cv2 as cv
import os
import sys
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from feature_extractor import *

from tqdm import tqdm
tqdm.pandas()

index = int(sys.argv[1])

data_csv_path = "./data/measurement/shanghai_all_dropna.csv"
img_dir_path = "./data/image/"

res = pd.read_csv(data_csv_path)
labels =    ['laplacian', 'variance', 'mean_r', 'mean_g', 'mean_b',
            'low_mean', 'high_mean', 'low_variance', 'high_variance'
            'dark_channel_mean', 'dark_channel_variance']

start = int(13895/3*index)
end = int(13895/3*(index+1))

generated_vectors = res[start:end]['filename'].progress_apply(vector_generator)
r = res[start:end]

for i in range(0, len(labels)):
    r[labels[i]] = [x[i] for x in generated_vectors]

r.to_csv("./data/measurement/shanghai_all_features_{}.csv".format(index))