#  Times-C

 Times-C is a GPU-accelerated  k-Shape time series clustering algorithm.

# Installation

### CUDA toolkit

To install CUDA toolkit please use [this link](https://developer.nvidia.com/cuda-downloads).

### Complier
It is recommended to use a g++ compiler that supports C++14 or above. 

### Dataset

Datasets can be obtained from the following websites:

[Welcome to the UCR Time Series Classification/Clustering Page](https://www.cs.ucr.edu/~eamonn/time_series_data/)

[Time Series Classification Website](http://www.timeseriesclassification.com/dataset.php)

### k-Shape & tslearn algorithm

k-Shape algorithm can be obtained from the following websites:

[https://github.com/TheDatumOrg/kshape-python]

tslearn algorithm can be obtained from the following websites:

[https://github.com/tslearn-team/tslearn]

## Times-C GPU
### Compilation

Compile the code using the following command:

```
cd Code
make
```

### Execution

Run the code using the following command:

```
cd bin
```

```
./test -d devNum /path/to/dataset k flag init_chose
```

Description of the arguments:

- -d devnum: Specify the device the code runs on
- /path/to/dataset : The relative or absolute path to the file containing time series
- k : Number of clusters
- flag: Whether the first column of the specified dataset is a label. If the first column is not a label, set it to 1, otherwise set it to 0. The first column of UCR database is a label, while the first column of UCR and UAE databases is not a label
- init_chose: When the value is 0, it represents random initialization of cluster indices. When the value is 1, it represents random initialization of cluster centers.
- Example: `./test -d 0 ../data/Part_of_HandOutlines 2 0 0`

## Times-C CPU
### Execution
```
cd Code/test
```

```
python test_timesC.py /path/to/dataset k flag 
```

Description of the arguments:

- /path/to/dataset : The relative or absolute path to the file containing time series(Suggest using absolute paths)
- k : Number of clusters
- flag: Whether the first column of the specified dataset is a label. If the first column is not a label, set it to 1, otherwise set it to 0. The first column of UCR database is a label, while the first column of UCR and UAE databases is not a label

