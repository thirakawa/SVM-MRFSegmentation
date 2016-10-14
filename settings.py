# -*- coding: utf-8 -*-

import numpy as np
import cv2

# global settings
work_dir   = "data/"
train_file = "train.txt"
test_file  = "test.txt"

# file extensions
bsift_ext  = "_bsift.npy"
blab_ext   = "_blab.npy"
vw_ext     = "_bow.npy"
prob_ext   = "_prob.npy"
result_ext = "_res.png"

# blob settings
blob_size = 10
rect_size = 50

# grid SIFT settings
sift_scale = 5
sift_step  = 5

# kmeans settings
n_cluster   = 128
n_jobs      = -1
n_subsample = 400

# svm parameters
# None: grid search
svm_penarty        = None
svm_kernel         = 'linear'
gridsearch_verbose = True

# MRF parameter
mrf_pairwise_scale = 0.9


def erase_new_line(s):
    return s.strip()


def get_train_basename():
    f = open(work_dir + train_file)
    train_base = map(erase_new_line, f.readlines())
    f.close()
    return train_base


def get_test_basename():
    f = open(work_dir + test_file)
    test_base = map(erase_new_line, f.readlines())
    f.close()
    return test_base


def write_blob_wise_settings():
    base = get_train_basename()[0]
    img = cv2.imread("%simg/%s.bmp" % (work_dir, base), 0)
    ny, nx = img.shape
    x_stride = np.arange(0, nx - rect_size + 1, blob_size)
    y_stride = np.arange(0, ny - rect_size + 1, blob_size)
    b_size = int((rect_size - blob_size) / 2)

    # write
    f = open("%sblob_info.txt" % work_dir, 'w')
    f.write("img_size:%d %d\n" % (ny, nx))
    f.write("grid_size:%d %d\n" % (y_stride.shape[0], x_stride.shape[0]))
    f.write("boundary:%d\n" %  b_size)
    f.close()


def load_blob_wise_settings():
    f = open("%sblob_info.txt" % work_dir)
    lines = map(erase_new_line, f.readlines())
    img_size = map(int, lines[0].split(':')[1].split(' '))
    blob_size = map(int, lines[1].split(':')[1].split(' '))
    boundary_size = int(lines[2].split(':')[1])
    return img_size, blob_size, boundary_size
