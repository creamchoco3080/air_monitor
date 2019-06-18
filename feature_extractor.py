import pandas as pd
import numpy as np
import cv2 as cv
import os

img_dir_path = "./data/image/"

def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.
    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size
    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((7,7), (7,7), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch

def vector_generator(fname):
    img = cv.imread(os.path.join(img_dir_path, fname))
    
    dc = get_dark_channel(img, 15)
    dc_mean = cv.mean(dc)
    dc_var = np.var(dc)
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img1 = cv.GaussianBlur(img_gray, (3,3), 0)
    l = cv.Laplacian(img1, cv.CV_64F).var()

    v = np.var(img)
    m = cv.mean(img)

    fft = np.fft.fft2(img)
    d = 5
    low = np.copy(fft).real
    low[0:d, 0:d] = 0
    low_mean = cv.mean(low)[0]
    low_var = np.var(low)
    high = np.copy(fft).real
    high[d:, d:] = 0
    high_mean = cv.mean(high)[0]
    high_var = np.var(high)

    return [l, v, m[0], m[1], m[2], low_mean, high_mean, low_var, high_var, dc_mean[0], dc_var]