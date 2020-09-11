# Community detection for correlation matrices (spectral method)

Python implementation of the modified spectral method for community detection in correlation matrices (spec.py). In order to resolve subcommunities within communities the community detection algorithm can be recursively applied (multi.py), as proposed in the reference paper.

## References

* MacMahon, M. and Garlaschelli, D. (2015). Community Detection for Correlation Matrices.  
  Phys. Rev. X, 5(2), 021006. https://link.aps.org/doi/10.1103/PhysRevX.5.021006
  
* Zuo, Z. (2018). python-modularity-maximization  
  (Available at https://github.com/zhiyzuo/python-modularity-maximization)

## Requirements

* numpy

* matplotlib

## Notes and examples

Input files for spec.py and multi.py:  
* Each column should correspond to a time series  
* The header should contain the names of the vertices (time series)  

The file 'nyse_300.csv', which contains the temporal sequences of 300 stocks, can be taken as an example. In order to analyze this data set with spec.py, one could write:  

```
import spec

# The name of the file 
filename = "nyse_300.csv"

# The string used to separate values
delimiter = ","

# The desired null model (1, 2 or 3 according to the notation used in the reference paper)
l = 3

foo = spec.partition(filename, delimiter, l)
```

The same procedure can be applied to multi.py:  

```
import multi

# The name of the file
filename = "nyse_300.csv"

# The string used to separate values
delimiter = ","

# The desired null model (1, 2 or 3 according to the notation used in the reference paper)
l = 3

foo = multi.multiresolution_detect(filename, delimiter, l)
```

