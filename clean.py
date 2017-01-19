# -*- coding: utf-8 -*-

import os
import glob
import settings as st


def remove_files(filename):
    files = glob.glob(filename + "*")
    if len(files) > 0:
        for f in files:
            os.remove(f)


if __name__ == '__main__':

    remove_files("%strain_blob_info.txt" % st.work_dir)
    remove_files("%stest_blob_info.txt" % st.work_dir)

    remove_files("%skmeans.model*" % st.work_dir)
    remove_files("%sscaler.model*" % st.work_dir)
    remove_files("%ssvm.model*" % st.work_dir)

    remove_files("%simg/*%s" % (st.work_dir, st.bsift_ext))
    remove_files("%simg/*%s" % (st.work_dir, st.blab_ext))
    remove_files("%simg/*%s" % (st.work_dir, st.vw_ext))
    remove_files("%simg/*%s" % (st.work_dir, st.prob_ext))
    remove_files("%simg/*%s" % (st.work_dir, st.result_ext))
