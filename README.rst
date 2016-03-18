=======
 SVM-Python
=======

By Dongfeng Gu (https://www.gdf.name)

--------------
 Introduction
--------------

This is a basic implementation of a soft-margin kernel SVM solver in
Python using `numpy` and `cvxopt`.


--------------
 Usage
--------------

Training:
1. Go to the `bin/svm-train`, train the dataset first by passing the
dimension of the X features and the directory of the dataset

2. run the program and it will generate a `model.txt` which will be used
in the testing part

Testing:
1.  Go to the `bin/svm-test`, test the dataset first by passing the
dimension of the X features and the directory of the dataset, it will
automatically search for the `model.txt` file.

2. run the program and it will generate the error rate
