# -*- coding: utf-8 -*-

import numpy as np
import cv2

import settings as st


def blob_wise_sift(filename):
    img = cv2.imread(filename, 0)

    ny, nx = img.shape
    x_stride = np.arange(0, nx - st.rect_size + 1, st.blob_size)
    y_stride = np.arange(0, ny - st.rect_size + 1, st.blob_size)
    # print x_stride
    # print y_stride

    # SIFT
    sift = cv2.SIFT()
    dense = cv2.FeatureDetector_create("Dense")
    dense.setDouble('initFeatureScale', st.sift_scale)
    dense.setInt('initXyStep', st.sift_step)

    kp = dense.detect(img)
    kp, descriptors = sift.compute(img, kp)

    keypoints = np.array([k.pt for k in kp])
    kp_x = keypoints[:, 0]
    kp_y = keypoints[:, 1]
    # print keypoints  # [x, y]
    # print kp_x
    # print kp_y

    sift_list = []
    for y in y_stride:
        for x in x_stride:
            x_ind = np.logical_and(kp_x >= x, kp_x < x + st.rect_size)
            y_ind = np.logical_and(kp_y >= y, kp_y < y + st.rect_size)
            sift_list.append(descriptors[np.logical_and(x_ind, y_ind), :])

    return np.array(sift_list)


def blob_wise_label(filename):
    ref = np.load(filename)

    ny, nx = ref.shape
    x_stride = np.arange(0, nx - st.rect_size + 1, st.blob_size)
    y_stride = np.arange(0, ny - st.rect_size + 1, st.blob_size)

    label_list = []
    for y in y_stride:
        for x in x_stride:
            blob = ref[y:y+st.rect_size, x:x+st.rect_size]
            labels, counts = np.unique( blob, return_counts=True )
            label_list.append( int( labels[np.argmax(counts)] ) )
    return np.array(label_list)


if __name__ == '__main__':

    """
    extract blob-wise SIFT descriptors and labels
    """

    # compute blob-wise size settings
    st.write_blob_size_property()

    train_base = st.get_train_basename()
    test_base = st.get_test_basename()

    print "extract grid SIFT ......"
    print "    train data"
    for base in train_base:
        s_arr = blob_wise_sift("%simg/%s.bmp" % (st.work_dir, base))
        l_arr = blob_wise_label("%simg/%s.npy" % (st.work_dir, base))
        # cast
        s_arr = s_arr.astype(np.int16)
        l_arr = l_arr.astype(np.int8)

        np.save("%simg/%s%s" % (st.work_dir, base, st.bsift_ext), s_arr)
        np.save("%simg/%s%s" % (st.work_dir, base, st.blab_ext), l_arr)

    print "    test data"
    for base in test_base:
        s_arr = blob_wise_sift("%simg/%s.bmp" % (st.work_dir, base))
        l_arr = blob_wise_label("%simg/%s.npy" % (st.work_dir, base))
        # cast
        s_arr = s_arr.astype(np.int16)
        l_arr = l_arr.astype(np.int8)

        np.save("%simg/%s%s" % (st.work_dir, base, st.bsift_ext), s_arr)
        np.save("%simg/%s%s" % (st.work_dir, base, st.blab_ext), l_arr)

    print "extract grid SIFT; done.\n"
