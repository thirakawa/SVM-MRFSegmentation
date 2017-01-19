# SVM-MRF segmentation

This repository includes SVM-MRF segmentation [1].
Negative log values of posterior probabilities obtained from SVM classifier is used for data terms of MRF model.
For pairwise term, we used constant values range (0, 1) in order to make a segmentation result smooth.


## Execution enviroment
* Language: Python 2.7 (Anaconda 2.4.*)
* Modules: Numpy, OpenCV, scikit-learn, PyMaxFlow

## Usage
#### Paramters
All of paramters and directory for dataset are detenoed in settings.py.

#### Execution
    $ sh run.sh

## References
1. T. Hirakawa, T. Tamaki, B. Raytchev, K. Kaneda, T. Koide, Y. Kominami, S. Yoshida, S. Tanaka, "SVM-MRF segmentation of colorectal NBI endoscopic images," In Proc. of the IEEE International Conference of Engineering in Medicine and Biology Society (EMBC2014), pp.4739-4742, (Aug. 2014).
