#!/bin/bash


if test $# -eq 0; then
    echo "execute SVM-MRF segmentation scripts ....."
    python blob_wise_sift.py
    python visual_word.py
    python train_svm.py
    python mrf.py
else
    if test $1 = 'clean'; then
        echo "clean cache files ......"
        python clean.py
    fi
fi
