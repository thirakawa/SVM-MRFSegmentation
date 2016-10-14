# -*- coding: utf-8 -*-

import time
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

import settings as st


def load_train_vw_hist(filenames):
    vw_list = []
    for f in filenames:
        vw_list.extend(np.load(f))
    return np.array(vw_list)


def load_test_vw_hist(filenames):
    vw_list = []
    for f in filenames:
        vw_list.append(np.load(f))
    return vw_list

def train_svm(features, labels):
    print "train SVM ......"

    # scaling features
    print "    scaling"
    scaler = StandardScaler().fit(features)
    scaled_features = scaler.transform(features)

    # svm train
    start = time.time()
    if st.svm_penarty is None:
        print "    grid search"
        tuned_params = [{'kernel': ['linear'],
                             'C': [1e-1, 1e0, 1e1, 1e2, 1e3]}]
        grid_clf = GridSearchCV(SVC(probability=True), tuned_params, cv=5, n_jobs=-1)
        grid_clf.fit(scaled_features, labels)

        if st.gridsearch_verbose:
            print "\n    Grid scores on development set:"
            for params, mean_score, scores in grid_clf.grid_scores_:
                print "    %0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()*2, params)
            print ""

        print "    best parameter:", grid_clf.best_params_
        clf = grid_clf.best_estimator_
    else:
        print "    used parameter: kernel: %s, C: %f" % (st.svm_kernel, st.svm_penarty)
        clf = SVC(kernel=st.svm_kernel, C=st.svm_penarty, probability=True)
        clf.fit(scaled_features, labels)

    end = time.time()
    print "    train SVM; done."
    print "    time:", end - start, "\n"
    return clf, scaler


def blob_wise_clf(clf, scaler, features):
    scaled_features = scaler.transform(features)
    pred = clf.predict_proba(scaled_features)
    return pred


if __name__ == '__main__':

    """
    train SVM
    """

    # train files
    train_base = st.get_train_basename()
    train_feature_files = ["%simg/%s%s" % (st.work_dir, i, st.vw_ext) for i in train_base]
    train_features = load_train_vw_hist(train_feature_files)
    train_features_fortest = load_test_vw_hist(train_feature_files)

    train_label_files = ["%simg/%s%s" % (st.work_dir, i, st.blab_ext) for i in train_base]
    train_labels = load_train_vw_hist(train_label_files)

    # test files
    test_base = st.get_test_basename()
    test_feature_files = ["%simg/%s%s" % (st.work_dir, i, st.vw_ext) for i in test_base]
    test_features = load_test_vw_hist(test_feature_files)

    test_label_files = ["%simg/%s%s" % (st.work_dir, i, st.blab_ext) for i in test_base]
    test_labels = load_test_vw_hist(train_label_files)

    # train SVM
    clf, scaler = train_svm(train_features, train_labels)

    # classify test samples
    print "classify all samples ....."
    print "    train samples"
    for base, fe in zip(train_base, train_features_fortest):
        pred_prob = blob_wise_clf(clf, scaler, fe)
        np.save("%simg/%s%s" % (st.work_dir, base, st.prob_ext), pred_prob)

    print "    test samples"
    for base, fe in zip(test_base, test_features):
        pred_prob = blob_wise_clf(clf, scaler, fe)
        np.save("%simg/%s%s" % (st.work_dir, base, st.prob_ext), pred_prob)
    print "classify test samples; done.\n"

    # output clf and scaler
    joblib.dump(clf, "%ssvm.model" % (st.work_dir))
    joblib.dump(scaler, "%sscaler.model" % (st.work_dir))

