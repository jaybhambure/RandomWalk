# 1d random walker
```
1. compile and run 'randomWalkers.cpp' file, which will output 'walker_data.h5'. Then use the python file 'rw_plot.py' to visualize the initial and final data of positions of the random walkers.

mpicxx -o randomWalkers randomWalkers.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./randomWalkers
```


# Rough file to work with
```
mpicxx -o delete delete.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./delete
```
