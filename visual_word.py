# -*- coding: utf-8 -*-

import time
import numpy as np

from sklearn.cluster import KMeans
from sklearn.externals import joblib

import settings as st


def load_blob_wise_sift(files):
    all_feature_list = []
    for f in files:
        all_feature_list.append(np.load(f).astype(np.float64))
    return all_feature_list


def feature_wise_flatten(feature_list):
    reshaped_feature_list = []
    for f in feature_list:
        dim_feature = f.shape[2]
        reshaped_feature_list.extend( f.reshape((-1, dim_feature)).copy() )
    return np.array(reshaped_feature_list)


def kmeans_clustering(features):
    print "clustering ......."

    # subsample feature vectors
    print "    subsampling"
    if st.n_subsample > 1:
        n_feature = features.shape[0]
        subsampled_n_feature = int(n_feature / st.n_subsample)
        shuffled_index = np.random.permutation(n_feature)
        subsampled_list = []
        for si in shuffled_index[0:subsampled_n_feature]:
            subsampled_list.append(features[si, :].copy())
        subsampled_features = np.array(subsampled_list)
    else:
        subsampled_features = features.copy()

    # kmeans clustering
    print "    number of features:", subsampled_features.shape[0]
    start = time.time()
    kmeans = KMeans(n_clusters=st.n_cluster, n_jobs=st.n_jobs)
    kmeans.fit(subsampled_features)
    end = time.time()
    print "    clustering; done."
    print "    time:", end - start, "\n"

    return kmeans


def visual_word_histogram(kmeans, descriptors):
    # compute cluster of each descriptor
    clst = kmeans.predict(descriptors)
    # count the number of samples for each cluster
    c_ind, counts = np.unique(clst, return_counts=True)
    # visual word histogram
    bow = np.zeros(kmeans.cluster_centers_.shape[0])
    bow[c_ind] = counts
    # normalization
    bow /= np.linalg.norm(bow)
    return bow


def create_blob_wise_vw_hist(kmeans, feature_arr):
    bow_list = []
    for f in feature_arr:
        bow_list.append(visual_word_histogram(kmeans, f))
    return np.array(bow_list)


if __name__ == '__main__':

    """
    kmeans clustering and making visual word histograms
    """

    # train files
    train_base = st.get_train_basename()
    train_files = ["%simg/%s%s" % (st.work_dir, i, st.bsift_ext) for i in train_base]
    train_des = load_blob_wise_sift(train_files)
    train_des_arr = feature_wise_flatten(train_des)

    # test files
    test_base = st.get_test_basename()
    test_files = ["%simg/%s%s" % (st.work_dir, i, st.bsift_ext) for i in test_base]
    test_des = load_blob_wise_sift(test_files)

    # kmeans clustering
    km = kmeans_clustering(train_des_arr)

    # making visual word histograms
    print "making visual word histograms ........"
    print "    train data"
    for base, de in zip(train_base, train_des):
        bow_arr = create_blob_wise_vw_hist(km, de)
        np.save("%simg/%s%s" % (st.work_dir, base, st.vw_ext), bow_arr)

    print "    test data"
    for base, de in zip(test_base, test_des):
        bow_arr = create_blob_wise_vw_hist(km, de)
        np.save("%simg/%s%s" % (st.work_dir, base, st.vw_ext), bow_arr)
    print "making visual word histograms; done.\n"

    # output files
    joblib.dump(km, "%skmeans.model" % (st.work_dir))

