# -*- coding: utf-8 -*-

import numpy as np
import maxflow
import cv2

import settings as st


def load_prob_files(filenames):
    prob_list = []
    for f in filenames:
        prob_list.append(np.load(f))
    return prob_list


def mrf_estimate(prob, i_size, b_size, bound_size):

    # negative log
    neg_prob   = prob[:, 0]
    pos_prob   = prob[:, 1]
    neg_energy = -np.log(neg_prob).reshape(b_size)
    pos_energy = -np.log(pos_prob).reshape(b_size)

    # MAP estimate
    g       = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(b_size)
    g.add_grid_edges(nodeids, st.mrf_pairwise_scale)
    g.add_grid_tedges(nodeids, pos_energy, neg_energy)
    g.maxflow()
    sgm = g.get_grid_segments(nodeids)

    # cololize result
    res        = np.zeros((i_size[0], i_size[1], 3), dtype='uint8')
    expand_sgm = np.repeat(sgm, st.blob_size, axis=0).repeat(st.blob_size, axis=1)
    labels     = np.ones(i_size, dtype='int') * -1
    labels[bound_size:-bound_size, bound_size:-bound_size] = expand_sgm.astype('int')
    res[labels == 0, 0] = 255
    res[labels == 1, 2] = 255

    return res


if __name__ == '__main__':

    train_img_size, train_blob_size, test_img_size, test_blob_size = st.load_blob_size_property()

    print "load probability files ........"
    # print "    train data"
    # train_base = st.get_train_basename()
    # trian_prob_files = ["%simg/%s%s" % (st.work_dir, i, st.prob_ext) for i in train_base]
    # train_prob_list = load_prob_files(trian_prob_files)

    print "    test data"
    test_base = st.get_test_basename()
    test_prob_files = ["%simg/%s%s" % (st.work_dir, i, st.prob_ext) for i in test_base]
    test_prob_list = load_prob_files(test_prob_files)
    print "load probability files; done."
    print ""

    print "MAP estimate on a MRF model ........."
    # print "    train data"
    # for b, p in zip(train_base, train_prob_list):
    #     res = mrf_estimate(p, train_img_size[b], \
    #                        train_blob_size[b], st.boundary_size())
    #     cv2.imwrite("%simg/%s%s" % (st.work_dir, b, st.result_ext), res)

    print "    test data"
    for b, p in zip(test_base, test_prob_list):
        res = mrf_estimate(p, test_img_size[b], \
                           test_blob_size[b], st.boundary_size())
        cv2.imwrite("%simg/%s%s" % (st.work_dir, b, st.result_ext), res)
    print "MAP estimate; done."

