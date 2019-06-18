import pandas as pd
import numpy as np
import cv2 as cv
import os
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from feature_extractor import *

from tqdm import tqdm
tqdm.pandas()

data_csv_path = "./data/measurement/shanghai_all_dropna.csv"
img_dir_path = "./data/image/"

res = pd.read_csv(data_csv_path)

# t1 = cv.getTickCount()
# generated_vectors = res[:20]['filename'].progress_apply(vector_generator)
# t2 = cv.getTickCount()
# print((t2-t1)/cv.getTickFrequency())

pool = ThreadPool(8)
t1 = cv.getTickCount()
result = pool.map(vector_generator, res[:20]['filename'])
t2 = cv.getTickCount()
pool.close()
pool.join()
print((t2-t1)/cv.getTickFrequency())

# generated_vectors = res[:5]['filename'].progress_apply(vector_generator)

# res['laplacian'] = [x[0] for x in generated_vectors]
# res['variance'] = [x[1] for x in generated_vectors]
# res['mean_r'] = [x[2] for x in generated_vectors]
# res['mean_g'] = [x[3] for x in generated_vectors]
# res['mean_b'] = [x[4] for x in generated_vectors]
# res['low_mean'] = [x[5] for x in generated_vectors]
# res['high_mean'] = [x[6] for x in generated_vectors]
# res['low_variance'] = [x[7] for x in generated_vectors]
# res['high_variance'] = [x[8] for x in generated_vectors]
# res['dark_channel_mean'] = [x[9] for x in generated_vectors]
# res['dark_channel_variance'] = [x[10] for x in generated_vectors]

# res.to_csv("./data/measurement/shanghai_all_features.csv")