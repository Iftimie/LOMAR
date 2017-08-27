

The zip archive contains the following files and folders:

1  - folder with the data in Matlab *mat format from University Building 1; there is a file called train.mat including the training data
and several files (all others , called test*) including track data. There is also a file test_all.mat which merged all the tracks into 
a single track file

2  - folder with the data in Matlab *mat format from University Building 2; there is a file called train.mat including the training data
and several files (all others) including track data

affinity.py   Python file for clustering in RSS dimension via affinity propagation method

kmeans.py    Python file for clustering in 3D dimension via K means method


The files have been tested with Python 2.7 on Linux. The following packages are needed (in brackets the version that was used for testing):

numpy (>=1.7), 
scikit-learn (>=0.17)
scipy (>=0.13)


