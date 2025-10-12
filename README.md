# RandomWalk
# 1d random walker
```
mpicxx -o randomWalkers randomWalkers.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./randomWalkers
```


# Rough file to work with
```
mpicxx -o delete delete.cpp `pkg-config --cflags petsc` `pkg-config --libs petsc`
mpiexec -n 1 ./delete
```
